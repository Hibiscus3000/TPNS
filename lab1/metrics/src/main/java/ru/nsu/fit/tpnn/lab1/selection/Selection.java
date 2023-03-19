package ru.nsu.fit.tpnn.lab1.selection;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Selection {

    private final String valueName;
    private final String unitName;

    private final Map<Integer, Value> values = new HashMap<>(); // id => value
    private final Map<Integer, Interval> intervals = new HashMap<>();
    private BigDecimal min;
    private BigDecimal max;

    private BigDecimal entropy;
    private static final MathContext mathContext = new MathContext(7, RoundingMode.HALF_EVEN);

    private int numberOfNoDataEntries;

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
            min = value;
        }
        if (null == max || 0 < value.compareTo(max)) {
            max = value;
        }
    }

    private BigDecimal sampleMean;

    public BigDecimal getSampleMean() {
        if (null == sampleMean) {
            sampleMean = new BigDecimal(0, mathContext);
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
            sampleVariance = new BigDecimal(0, mathContext);
            BigDecimal sampleMean = getSampleMean();
            for (Map.Entry<Integer, Value> valueEntry : values.entrySet().stream().toList()) {
                sampleVariance =
                        sampleVariance.add(valueEntry.getValue().getValue().subtract(sampleMean).pow(2));
            }
            sampleVariance = sampleVariance.divide(BigDecimal.valueOf(values.size()), mathContext);
        }
        return sampleVariance;
    }

    public BigDecimal getEntropy() {
        if (null == entropy) {
            splitIntervals();
            entropy = new BigDecimal(0, mathContext);
            for (Map.Entry<Integer, Interval> intervalEntry : intervals.entrySet().stream().toList()) {
                BigDecimal frequency = new BigDecimal(intervalEntry.getValue().getNumberOfValues())
                        .divide(new BigDecimal(values.size()), mathContext);
                if (!frequency.equals(BigDecimal.valueOf(0))) {
                    entropy = entropy.subtract(frequency.multiply(log2(frequency), mathContext));
                }
            }
        }
        BigDecimal noDataFrequency = BigDecimal.valueOf(numberOfNoDataEntries)
                .divide(BigDecimal.valueOf(values.size()), mathContext);
        return entropy.subtract(log2(noDataFrequency).multiply(noDataFrequency, mathContext));
    }

    private void splitIntervals() { //Sturgess rule
        int numberOfIntervals = (int) Math.ceil(log2(values.size()));
        BigDecimal delta = max.subtract(min).divide(BigDecimal.valueOf(numberOfIntervals), mathContext);
        BigDecimal leftBorder = new BigDecimal(min.doubleValue(), mathContext);
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

    private BigDecimal getNormalizedConditionalEntropy(Selection onWhichDepend) {
        BigDecimal normalizedConditionalEntropy = new BigDecimal(0, mathContext);
        for (Map.Entry<Integer, Interval> intervalEntry : onWhichDepend.intervals.entrySet().stream().toList()) {
            int numberOfValuesTotal = 0;
            int numberOfIntervals = intervals.size();
            int[] frequencies = new int[numberOfIntervals + 1]; //frequencies[intervals.size()] - no data frequency
            for (Integer valueId : intervalEntry.getValue().getValuesIds()) {
                Value value = values.get(valueId);
                if (null != value) {
                    ++frequencies[value.getIntervalId()];
                } else {
                    ++frequencies[numberOfIntervals];
                }
                ++numberOfValuesTotal;
            }
            for (int frequency : frequencies) {
                if (0 == frequency) {
                    continue;
                }
                normalizedConditionalEntropy = normalizedConditionalEntropy
                        .subtract(BigDecimal
                                .valueOf(log2(((double) frequency) / numberOfValuesTotal))
                                .multiply(BigDecimal
                                        .valueOf(((double) frequency) / onWhichDepend.values.size())));
            }
        }
        return normalizedConditionalEntropy;
    }

    private BigDecimal getInfoGain(Selection onWhichDepend) {
        return getEntropy().subtract(getNormalizedConditionalEntropy(onWhichDepend));
    }

    public BigDecimal getGainRatio(Selection onWhichDepend) {
        return getInfoGain(onWhichDepend)
                .divide(onWhichDepend.getEntropy()
                        .multiply(BigDecimal.valueOf(-1)), mathContext);
    }

    public BigDecimal getCorrelation(Selection other) {
        Selection productSelection = new Selection(String.format("product(%s,%s)", valueName,
                other.valueName), String.format("(%s,%s)", unitName, other.unitName));
        int id = 0;
        for (Map.Entry<Integer, Value> thisValueEntry : values.entrySet().stream().toList()) {
            Value value = other.values.get(thisValueEntry.getKey());
            if (null != value) {
                productSelection.setNextValue(id++,
                        thisValueEntry.getValue().getValue()
                                .multiply(value.getValue()));
            }
        }

        BigDecimal productSampleMean = getSampleMean().multiply(other.getSampleMean(), mathContext);
        BigDecimal covariance = productSelection.getSampleMean().subtract(productSampleMean, mathContext);
        BigDecimal variancesProduct = getSampleVariance().multiply(other.getSampleVariance(),
                        mathContext)
                .sqrt(mathContext);
        try {
            return covariance.divide(variancesProduct, mathContext);
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
        return BigDecimal.valueOf(Math.log10(value.doubleValue()) / Math.log10(2));
    }

    public void setNumberOfNoDataEntries(int numberOfEntries) {
        numberOfNoDataEntries = numberOfEntries - values.size();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Selection selection = (Selection) o;
        return Objects.equals(valueName, selection.valueName) && Objects.equals(unitName, selection.unitName) && Objects.equals(values, selection.values) && Objects.equals(intervals, selection.intervals) && Objects.equals(min, selection.min) && Objects.equals(max, selection.max) && Objects.equals(entropy, selection.entropy) && Objects.equals(sampleMean, selection.sampleMean) && Objects.equals(sampleVariance, selection.sampleVariance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(valueName, unitName, values, intervals, min, max, entropy, sampleMean, sampleVariance);
    }
}
