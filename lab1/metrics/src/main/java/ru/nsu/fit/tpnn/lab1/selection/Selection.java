package ru.nsu.fit.tpnn.lab1.selection;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.*;

import static ru.nsu.fit.tpnn.lab1.selection.Value.roundingMode;
import static ru.nsu.fit.tpnn.lab1.selection.Value.scale;

public class Selection {

    private final String valueName;
    private final String unitName;

    private final Map<Integer, Value> values = new HashMap<>(); // id => value
    private final Map<Integer, Interval> intervals = new HashMap<>();
    private BigDecimal min;
    private BigDecimal max;

    private static final MathContext mathContext = new MathContext(10, RoundingMode.HALF_EVEN);

    private final List<Integer> noValueEntries = new ArrayList<>();

    public Selection(String valueName, String unitName) {
        this.valueName = valueName;
        this.unitName = unitName;
    }

    public void setNextValue(int id, double value) {
        BigDecimal newValue = new BigDecimal(value, mathContext);
        setNextValue(id, newValue);
    }

    public void setNextValue(int id, BigDecimal value) {
        values.put(id, new Value(value));
        if (null == min || 0 > value.compareTo(min)) {
            min = value.setScale(scale, roundingMode);
        }
        if (null == max || 0 < value.compareTo(max)) {
            max = value.setScale(scale, roundingMode);
        }
    }

    public void setNoDataEntries(int rowsTotal) {
        for (int i = 0; i < rowsTotal; ++i) {
            if (null == values.get(i)) {
                noValueEntries.add(i);
            }
        }
    }

    private BigDecimal sampleMean;

    public BigDecimal getSampleMean() {
        if (null == sampleMean) {
            sampleMean = new BigDecimal(0, mathContext).setScale(2, roundingMode);
            for (Map.Entry<Integer, Value> value : values.entrySet().stream().toList()) {
                sampleMean = sampleMean.add(value.getValue().getValue());
            }
            sampleMean = sampleMean.divide(BigDecimal.valueOf(values.size()), mathContext);
        }
        return sampleMean;
    }

    private BigDecimal sampleVariance;

    public BigDecimal getSampleVariance() {
        if (null == sampleVariance) {
            sampleVariance = new BigDecimal(0, mathContext).setScale(scale, roundingMode);
            BigDecimal sampleMean = getSampleMean();
            for (Map.Entry<Integer, Value> valueEntry : values.entrySet().stream().toList()) {
                sampleVariance =
                        sampleVariance.add(valueEntry.getValue().getValue().subtract(sampleMean).pow(2));
            }
        }
        return sampleVariance;
    }

    public BigDecimal getEntropy(boolean withNoData) {
        splitIntervals();
        BigDecimal entropy = new BigDecimal(0, mathContext).setScale(scale, roundingMode);
        int numberOfValues = values.size();
        int numberOfNoValuesEntries = noValueEntries.size();
        for (Map.Entry<Integer, Interval> intervalEntry : intervals.entrySet().stream().toList()) {
            BigDecimal frequency = new BigDecimal(intervalEntry.getValue().getNumberOfValues())
                    .divide(new BigDecimal(numberOfValues + (withNoData ? numberOfNoValuesEntries : 0)),
                            mathContext).setScale(scale, roundingMode);
            if (!frequency.equals(BigDecimal.valueOf(0).setScale(scale, roundingMode))) {
                entropy = entropy.subtract(frequency.multiply(log2(frequency), mathContext))
                        .setScale(scale, roundingMode);
            }
        }
        if (withNoData && 0 < numberOfNoValuesEntries) {
            BigDecimal noDataFrequency = BigDecimal.valueOf(numberOfNoValuesEntries)
                    .divide(BigDecimal.valueOf(numberOfValues + numberOfNoValuesEntries),
                            mathContext).setScale(scale, roundingMode);
            entropy = entropy.subtract(log2(noDataFrequency).multiply(noDataFrequency, mathContext))
                    .setScale(scale, roundingMode);
        }
        return entropy;
    }

    private void splitIntervals() { //Sturgess rule
        int numberOfIntervals = (int) Math.ceil(log2(values.size()));
        BigDecimal delta = max.subtract(min).divide(BigDecimal.valueOf(numberOfIntervals), mathContext)
                .setScale(scale, roundingMode);
        BigDecimal leftBorder = new BigDecimal(min.doubleValue(), mathContext).setScale(scale, roundingMode);
        for (int i = 0; i < numberOfIntervals; ++i) {
            BigDecimal rightBorder = leftBorder.add(delta);
            Interval interval = new Interval(i, leftBorder, rightBorder, i == numberOfIntervals - 1);
            leftBorder = rightBorder;
            intervals.put(i, interval);
            for (Map.Entry<Integer, Value> valueEntry : values.entrySet().stream().toList()) {
                Value value = valueEntry.getValue();
                if (interval.isIn(value.getValue())) {
                    interval.addValue(valueEntry.getKey(), value);
                }
            }
        }
    }

    private BigDecimal getNormalizedConditionalEntropy(Selection onWhichDepend, boolean withNoData) {
        BigDecimal normalizedConditionalEntropy = new BigDecimal(0, mathContext)
                .setScale(scale, roundingMode);
        for (Map.Entry<Integer, Interval> intervalEntry : onWhichDepend.intervals.entrySet().stream().toList()) {
            normalizedConditionalEntropy = countInterval(intervalEntry.getValue().getValuesIds(),
                    normalizedConditionalEntropy, withNoData);
        }
        return withNoData ? countInterval(noValueEntries, normalizedConditionalEntropy,
                true)
                : normalizedConditionalEntropy;
    }

    private BigDecimal countInterval(List<Integer> valueIds, BigDecimal normalizedConditionalEntropy,
                                     boolean withNoData) {
        int numberOfValuesTotal = 0;
        int numberOfIntervals = intervals.size();
        int[] appearances = new int[numberOfIntervals + (withNoData ? 1 : 0)]; //frequencies[intervals.size()] - no data frequency
        for (Integer valueId : valueIds) {
            Value value = values.get(valueId);
            if (null != value) {
                ++appearances[value.getIntervalId()];
            } else {
                if (withNoData) {
                    ++appearances[numberOfIntervals];
                } else {
                    continue;
                }
            }
            ++numberOfValuesTotal;
        }
        for (int appearance : appearances) {
            if (0 == appearance) {
                continue;
            }
            normalizedConditionalEntropy = normalizedConditionalEntropy
                    .subtract(BigDecimal
                            .valueOf(log2(((double) appearance) / numberOfValuesTotal))
                            .multiply(BigDecimal
                                    .valueOf(((double) appearance) / (values.size() + (withNoData ? noValueEntries.size() : 0))))
                            .setScale(scale, roundingMode));
        }
        return normalizedConditionalEntropy;
    }

    private BigDecimal getInfoGain(Selection onWhichDepend, boolean withNoData) {
        return getEntropy(withNoData).subtract(getNormalizedConditionalEntropy(onWhichDepend, withNoData))
                .setScale(scale, roundingMode);
    }

    public BigDecimal getGainRatio(Selection onWhichDepend, boolean withNoData) {
        try {
            return getInfoGain(onWhichDepend, withNoData)
                    .divide(onWhichDepend.getEntropy(withNoData), mathContext)
                    .setScale(scale, roundingMode);
        } catch (ArithmeticException arithmeticException) {
            return BigDecimal.valueOf(0);
        }
    }

    public BigDecimal getCorrelation(Selection other) {
        BigDecimal covariance = BigDecimal.valueOf(0).setScale(scale, roundingMode);
        for (Map.Entry<Integer, Value> thisValueEntry : values.entrySet().stream().toList()) {
            Value otherValue = other.values.get(thisValueEntry.getKey());
            if (null != otherValue) {
                Value thisValue = thisValueEntry.getValue();
                covariance = covariance.add(thisValue.getValue()
                                .subtract(getSampleMean(), mathContext)
                                .multiply(otherValue.getValue()
                                        .subtract(other.getSampleMean(), mathContext), mathContext))
                        .setScale(scale, roundingMode);
            }
        }

        BigDecimal variancesProductSqrt = getSampleVariance().multiply(other.getSampleVariance(),
                        mathContext).setScale(scale, roundingMode).sqrt(mathContext)
                .setScale(scale, roundingMode);
        try {
            return covariance.divide(variancesProductSqrt, mathContext).setScale(scale, roundingMode);
        } catch (ArithmeticException arithmeticException) {
            return BigDecimal.valueOf(0);
        }
    }

    public String getValueName() {
        return String.format("%s(%s)", valueName, unitName);
    }

    public boolean isEmpty() {
        return 0 == values.size();
    }

    private double log2(double value) {
        return Math.log10(value) / Math.log10(2);
    }

    private BigDecimal log2(BigDecimal value) {
        return BigDecimal.valueOf(Math.log10(value.doubleValue()) / Math.log10(2)).setScale(scale, roundingMode);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Selection selection = (Selection) o;
        return Objects.equals(valueName, selection.valueName) && Objects.equals(unitName, selection.unitName) && Objects.equals(values, selection.values) && Objects.equals(intervals, selection.intervals) && Objects.equals(min, selection.min) && Objects.equals(max, selection.max) && Objects.equals(sampleMean, selection.sampleMean) && Objects.equals(sampleVariance, selection.sampleVariance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(valueName, unitName, values, intervals, min, max, sampleMean, sampleVariance);
    }
}
