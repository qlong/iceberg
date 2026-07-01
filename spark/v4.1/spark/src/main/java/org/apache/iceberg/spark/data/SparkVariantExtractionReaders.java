/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.spark.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.ZoneId;
import java.util.List;
import org.apache.iceberg.expressions.PathUtil;
import org.apache.iceberg.parquet.ParquetValueReader;
import org.apache.iceberg.parquet.ParquetVariantExtractionReaders;
import org.apache.iceberg.parquet.ParquetVariantExtractionReaders.VariantExtractionField;
import org.apache.iceberg.parquet.ParquetVariantExtractionReaders.VariantExtractionRow;
import org.apache.iceberg.parquet.ParquetVariantReaders.DelegatingValueReader;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.variants.PhysicalType;
import org.apache.iceberg.variants.VariantMetadata;
import org.apache.iceberg.variants.VariantValue;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.MessageType;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.expressions.variant.VariantCastArgs;
import org.apache.spark.sql.internal.SQLConf;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;
import org.apache.spark.unsafe.types.VariantVal;

/** Parquet readers that materialize Spark variant extraction struct rows from shredded variants. */
public class SparkVariantExtractionReaders {
  // Empty variant metadata dictionary, reused for delegating primitive-leaf casts (primitives do
  // not reference the dictionary, so the per-row dictionary need not be serialized).
  private static final byte[] EMPTY_VARIANT_METADATA = serializeMetadata(VariantMetadata.empty());

  private SparkVariantExtractionReaders() {}

  public static ParquetValueReader<InternalRow> buildStructReader(
      MessageType fileSchema,
      GroupType variantGroup,
      List<String> variantColumnPath,
      StructType extractionStruct) {
    List<VariantExtractionField> fields = Lists.newArrayList();
    int numFields = 0;
    for (StructField field : extractionStruct.fields()) {
      numFields = Math.max(numFields, Integer.parseInt(field.name()) + 1);
    }

    DataType[] targetTypes = new DataType[numFields];
    boolean[] failOnError = new boolean[numFields];
    String[] timeZoneIds = new String[numFields];
    for (StructField field : extractionStruct.fields()) {
      int ordinal = Integer.parseInt(field.name());
      targetTypes[ordinal] = field.dataType();
      failOnError[ordinal] = SparkVariantExtractionUtil.failOnError(field);
      timeZoneIds[ordinal] = SparkVariantExtractionUtil.timeZoneId(field);
      fields.add(
          new VariantExtractionField(
              ordinal,
              SparkVariantExtractionUtil.isPlaceholderExtraction(field),
              PathUtil.parse(SparkVariantExtractionUtil.extractionPath(field))));
    }

    ParquetValueReader<VariantExtractionRow> parquetReader =
        ParquetVariantExtractionReaders.buildRowReader(
            fileSchema, variantGroup, variantColumnPath, fields);

    return new SparkVariantExtractionStructReader(
        parquetReader, targetTypes, failOnError, timeZoneIds);
  }

  /** Counts leaf Parquet columns referenced by a reader tree. */
  public static int leafColumnCount(ParquetValueReader<?> reader) {
    return ParquetVariantExtractionReaders.leafColumnCount(reader);
  }

  /** Visible for unit tests in {@code org.apache.iceberg.spark.data}. */
  static Object toSparkValueForTests(VariantValue value, DataType targetType) {
    return toSparkValueForTests(value, targetType, false);
  }

  /** Visible for unit tests in {@code org.apache.iceberg.spark.data}. */
  static Object toSparkValueForTests(VariantValue value, DataType targetType, boolean failOnError) {
    // Tests pass standalone primitive values, so an empty metadata dictionary and UTC suffice.
    return toSparkValue(value, targetType, failOnError, VariantMetadata.empty(), "UTC");
  }

  private static class SparkVariantExtractionStructReader
      extends DelegatingValueReader<VariantExtractionRow, InternalRow> {
    private final DataType[] targetTypes;
    private final boolean[] failOnError;
    private final String[] timeZoneIds;

    private SparkVariantExtractionStructReader(
        ParquetValueReader<VariantExtractionRow> reader,
        DataType[] targetTypes,
        boolean[] failOnError,
        String[] timeZoneIds) {
      super(reader);
      this.targetTypes = targetTypes;
      this.failOnError = failOnError;
      this.timeZoneIds = timeZoneIds;
    }

    @Override
    public InternalRow read(InternalRow reuse) {
      VariantExtractionRow row = readFromDelegate(null);
      int numFields = row.numFields();
      GenericInternalRow result =
          reuse instanceof GenericInternalRow
              ? (GenericInternalRow) reuse
              : new GenericInternalRow(numFields);

      if (row.metadata() == null) {
        // SQL NULL variant: match Spark variant_get semantics by nulling every extraction slot,
        // including full-variant placeholder slots. Distinct from a non-null variant where only
        // missing or JSON-null paths produce per-field SQL NULLs.
        for (int i = 0; i < numFields; i += 1) {
          result.setNullAt(i);
        }
        return result;
      }

      for (int i = 0; i < numFields; i += 1) {
        if (row.placeholder(i)) {
          result.setBoolean(i, true);
        } else {
          Object sparkValue =
              toSparkValue(
                  row.value(i), targetTypes[i], failOnError[i], row.metadata(), timeZoneIds[i]);
          if (sparkValue == null) {
            result.setNullAt(i);
          } else {
            result.update(i, sparkValue);
          }
        }
      }

      return result;
    }
  }

  /**
   * Materializes a single extracted shredded value as the Spark {@code targetType}.
   *
   * <p>Uses a narrow inline conversion for the common, stable pairs ({@link #inlineOwned}) and
   * delegates everything else to Spark's {@code VariantGet.cast}. The inline path exists purely for
   * performance: it reads the shredded primitive directly, whereas delegating must serialize the
   * value back into a Spark {@code VariantVal} and run a {@code Cast} — on the order of ~1 KB of
   * transient allocation per value (vs. tens of bytes inline), which is significant on wide scans.
   * It is intentionally kept to pairs that are either trivially Spark-identical (identity unwraps,
   * integer range-checks) or that Spark's variant cannot represent at all and so <em>must</em> be
   * handled here ({@code TIME}, nanos timestamps). Every cross-type, lossy, or zone-sensitive cast
   * delegates, so its semantics — including {@code failOnError} throwing and {@code timeZoneId}
   * handling — match the non-pushdown {@code variant_get} path exactly. A cross-check test asserts
   * the inline pairs equal {@code VariantGet.cast} so the two paths cannot silently diverge.
   */
  private static Object toSparkValue(
      VariantValue value,
      DataType targetType,
      boolean failOnError,
      VariantMetadata metadata,
      String timeZoneId) {
    if (value == null || value.type() == PhysicalType.NULL) {
      // Missing path or variant null: always SQL NULL, regardless of failOnError. This mirrors
      // Spark's VariantGet, which returns null for a missing path / variant null before any cast.
      return null;
    }

    if (inlineOwned(value.type(), targetType)) {
      return inlineConvert(value, targetType, failOnError);
    }

    // Any other (sourceType, targetType): delegate to Spark's VariantGet.cast so cross-type and
    // lossy casts (e.g. string->long, double->float, decimal rescale, TZ<->NTZ) match the
    // non-pushdown variant_get path exactly, including throwing on failOnError.
    return delegate(value, metadata, targetType, failOnError, timeZoneId);
  }

  /**
   * Returns true when the selective reader converts this {@code (sourceType, targetType)} pair
   * directly, without delegating to Spark. The inline set is deliberately small and stable: pairs
   * that are the identity (nothing to get wrong), integer-family range-checks (stable ANSI
   * semantics), or physical types Spark's variant cannot represent ({@code TIME}, nanos timestamps)
   * and therefore must be handled here. Everything else delegates.
   */
  private static boolean inlineOwned(PhysicalType sourceType, DataType targetType) {
    if (isIntegerTarget(targetType)) {
      // Integer/TIME source -> integer target (identity, widening, or range-checked narrowing).
      // TIME has no Spark variant counterpart, so it must be inline.
      return isInlineIntegerSource(sourceType);
    } else if (DataTypes.StringType.sameType(targetType)) {
      return sourceType == PhysicalType.STRING || sourceType == PhysicalType.UUID;
    } else if (DataTypes.BinaryType.sameType(targetType)) {
      return sourceType == PhysicalType.BINARY;
    } else if (DataTypes.BooleanType.sameType(targetType)) {
      return sourceType == PhysicalType.BOOLEAN_TRUE || sourceType == PhysicalType.BOOLEAN_FALSE;
    } else if (DataTypes.DateType.sameType(targetType)) {
      return sourceType == PhysicalType.DATE;
    } else if (DataTypes.DoubleType.sameType(targetType)) {
      return sourceType == PhysicalType.DOUBLE;
    } else if (DataTypes.FloatType.sameType(targetType)) {
      return sourceType == PhysicalType.FLOAT;
    } else if (targetType instanceof TimestampType) {
      // Nanos timestamps must be inline: Spark's variant cannot read them.
      return sourceType == PhysicalType.TIMESTAMPTZ || sourceType == PhysicalType.TIMESTAMPTZ_NANOS;
    } else if (targetType instanceof TimestampNTZType) {
      return sourceType == PhysicalType.TIMESTAMPNTZ
          || sourceType == PhysicalType.TIMESTAMPNTZ_NANOS;
    }

    // DecimalType and any other target: delegate to Spark.
    return false;
  }

  private static boolean isIntegerTarget(DataType targetType) {
    return DataTypes.LongType.sameType(targetType)
        || DataTypes.IntegerType.sameType(targetType)
        || DataTypes.ShortType.sameType(targetType)
        || DataTypes.ByteType.sameType(targetType);
  }

  private static boolean isInlineIntegerSource(PhysicalType sourceType) {
    switch (sourceType) {
      case INT8:
      case INT16:
      case INT32:
      case INT64:
      case TIME:
        return true;
      default:
        return false;
    }
  }

  /**
   * Delegates the cast to Spark's own {@code VariantGet.cast} by serializing the extracted leaf
   * value into a Spark {@link VariantVal}. Used for every pair outside {@link #inlineOwned} so that
   * cross-type and lossy casts match the non-pushdown {@code variant_get} path exactly (returning
   * Spark's value, or throwing {@code INVALID_VARIANT_CAST} when {@code failOnError = true}).
   */
  private static Object delegate(
      VariantValue value,
      VariantMetadata metadata,
      DataType targetType,
      boolean failOnError,
      String timeZoneId) {
    byte[] metadataBytes;
    if (value.type() == PhysicalType.OBJECT || value.type() == PhysicalType.ARRAY) {
      // Object/array values reference the dictionary (e.g. for cast-to-string JSON rendering).
      metadataBytes = serializeMetadata(metadata != null ? metadata : VariantMetadata.empty());
    } else {
      metadataBytes = EMPTY_VARIANT_METADATA;
    }

    byte[] valueBytes = new byte[value.sizeInBytes()];
    value.writeTo(ByteBuffer.wrap(valueBytes).order(ByteOrder.LITTLE_ENDIAN), 0);

    // Fall back to the session time zone (matching the non-pushdown variant_get path) when the
    // extraction metadata did not carry one.
    String zone = timeZoneId != null ? timeZoneId : SQLConf.get().sessionLocalTimeZone();
    VariantCastArgs castArgs =
        new VariantCastArgs(failOnError, scala.Option.apply(zone), ZoneId.of(zone));
    return org.apache.spark.sql.catalyst.expressions.variant.VariantGet$.MODULE$.cast(
        new VariantVal(valueBytes, metadataBytes), targetType, castArgs);
  }

  private static byte[] serializeMetadata(VariantMetadata metadata) {
    byte[] bytes = new byte[metadata.sizeInBytes()];
    metadata.writeTo(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN), 0);
    return bytes;
  }

  private static Object inlineConvert(
      VariantValue value, DataType targetType, boolean failOnError) {
    if (DataTypes.StringType.sameType(targetType)) {
      String stringValue = asString(value);
      return stringValue != null ? UTF8String.fromString(stringValue) : null;
    } else if (DataTypes.LongType.sameType(targetType)) {
      return asLong(value);
    } else if (DataTypes.IntegerType.sameType(targetType)) {
      // Integer narrowing (long -> int/short/byte): read the shredded value as a long, truncate to
      // the narrower width, then detect overflow by round-tripping the narrowed value back to long
      // and checking equality. A mismatch means the value did not fit; overflow() then mirrors
      // Spark's Cast(long, <narrow>, EvalMode.TRY) -> invalidCast: throw INVALID_VARIANT_CAST when
      // failOnError, otherwise SQL NULL. This is done inline (rather than delegating to
      // VariantGet.cast) because narrowing is a common, hot extraction path and its semantics are
      // simple and provably identical to Spark's, so we avoid the per-value serialization cost.
      Long longValue = asLong(value);
      if (longValue == null) {
        return null;
      }
      int intValue = (int) (long) longValue;
      return (long) intValue == longValue ? intValue : overflow(value, targetType, failOnError);
    } else if (DataTypes.ByteType.sameType(targetType)) {
      Long longValue = asLong(value);
      if (longValue == null) {
        return null;
      }
      byte byteValue = (byte) (long) longValue;
      return (long) byteValue == longValue ? byteValue : overflow(value, targetType, failOnError);
    } else if (DataTypes.ShortType.sameType(targetType)) {
      Long longValue = asLong(value);
      if (longValue == null) {
        return null;
      }
      short shortValue = (short) (long) longValue;
      return (long) shortValue == longValue ? shortValue : overflow(value, targetType, failOnError);
    } else if (DataTypes.DateType.sameType(targetType)) {
      return asDateDays(value);
    } else if (targetType instanceof TimestampType) {
      return asTimestampMicros(value, false);
    } else if (targetType instanceof TimestampNTZType) {
      return asTimestampMicros(value, true);
    } else if (DataTypes.BinaryType.sameType(targetType)) {
      return asBinary(value);
    } else if (DataTypes.BooleanType.sameType(targetType)) {
      return asBoolean(value);
    } else if (DataTypes.DoubleType.sameType(targetType)) {
      return asDouble(value);
    } else if (DataTypes.FloatType.sameType(targetType)) {
      Double doubleValue = asDouble(value);
      return doubleValue != null ? (float) (double) doubleValue : null;
    }

    // Unreachable for inline-owned pairs; defensive null.
    return null;
  }

  /**
   * Handles an integer narrowing overflow: the shredded value is a valid integer for the wider type
   * but does not fit the requested narrower integer type. Spark's {@code variant_get} ({@code
   * failOnError = true}) throws {@code INVALID_VARIANT_CAST}; {@code try_variant_get} ({@code
   * failOnError = false}) returns SQL NULL. This mirrors that. Only integer narrowing reaches here;
   * all non-integer-family casts delegate to Spark.
   */
  private static Object overflow(VariantValue value, DataType targetType, boolean failOnError) {
    if (failOnError) {
      throw invalidVariantCast(value, targetType);
    }

    return null;
  }

  private static RuntimeException invalidVariantCast(VariantValue value, DataType targetType) {
    String display =
        value.type() == PhysicalType.OBJECT || value.type() == PhysicalType.ARRAY
            ? value.toString()
            : String.valueOf(value.asPrimitive().get());
    // Reuse Spark's own error factory so the pushed-down path raises the exact same
    // INVALID_VARIANT_CAST error the engine would raise on the whole-variant variant_get path.
    return (RuntimeException)
        org.apache.spark.sql.errors.QueryExecutionErrors$.MODULE$.invalidVariantCast(
            display, targetType);
  }

  private static String asString(VariantValue value) {
    switch (value.type()) {
      case STRING:
      case UUID:
        return (String) value.asPrimitive().get();
      default:
        // Variant physical type is not string-compatible; return null like variant_get does.
        return null;
    }
  }

  private static Long asLong(VariantValue value) {
    switch (value.type()) {
      case INT8:
      case INT16:
      case INT32:
      case INT64:
      case TIME:
        return ((Number) value.asPrimitive().get()).longValue();
      default:
        return null;
    }
  }

  private static Integer asDateDays(VariantValue value) {
    if (value.type() == PhysicalType.DATE) {
      return (Integer) value.asPrimitive().get();
    }

    return null;
  }

  private static Long asTimestampMicros(VariantValue value, boolean timestampNtz) {
    switch (value.type()) {
      case TIMESTAMPTZ:
      case TIMESTAMPTZ_NANOS:
        if (timestampNtz) {
          return null;
        }
        return toMicros(value);
      case TIMESTAMPNTZ:
      case TIMESTAMPNTZ_NANOS:
        if (!timestampNtz) {
          return null;
        }
        return toMicros(value);
      default:
        return null;
    }
  }

  private static Long toMicros(VariantValue value) {
    long raw = ((Number) value.asPrimitive().get()).longValue();
    switch (value.type()) {
      case TIMESTAMPTZ_NANOS:
      case TIMESTAMPNTZ_NANOS:
        return raw / 1000L;
      case TIMESTAMPTZ:
      case TIMESTAMPNTZ:
        return raw;
      default:
        return null;
    }
  }

  private static byte[] asBinary(VariantValue value) {
    if (value.type() != PhysicalType.BINARY) {
      return null;
    }

    ByteBuffer buffer = (ByteBuffer) value.asPrimitive().get();
    byte[] bytes = new byte[buffer.remaining()];
    buffer.duplicate().get(bytes);
    return bytes;
  }

  private static Double asDouble(VariantValue value) {
    switch (value.type()) {
      case FLOAT:
      case DOUBLE:
      case INT8:
      case INT16:
      case INT32:
      case INT64:
        return ((Number) value.asPrimitive().get()).doubleValue();
      default:
        return null;
    }
  }

  private static Boolean asBoolean(VariantValue value) {
    switch (value.type()) {
      case BOOLEAN_TRUE:
        return true;
      case BOOLEAN_FALSE:
        return false;
      default:
        return null;
    }
  }
}
