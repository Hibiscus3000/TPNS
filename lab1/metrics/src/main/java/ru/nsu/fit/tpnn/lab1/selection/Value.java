package ru.nsu.fit.tpnn.lab1.selection;

import java.math.BigDecimal;
import java.math.RoundingMode;

import static java.math.RoundingMode.HALF_EVEN;

public class Value {

    public static final int scale = 8;
    public static final RoundingMode roundingMode = HALF_EVEN;

    private final BigDecimal value;
    private Integer intervalId;

    public Value(BigDecimal value) {
        this.value = value.setScale(scale, roundingMode);
    }

    public BigDecimal getValue() {
        return value;
    }

    public void setIntervalId(Integer id) {
        this.intervalId = id;
    }

    public Integer getIntervalId() {
        return intervalId;
    }
}
