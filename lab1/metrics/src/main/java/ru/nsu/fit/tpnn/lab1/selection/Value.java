package ru.nsu.fit.tpnn.lab1.selection;

import java.math.BigDecimal;

public class Value {

    private final BigDecimal value;
    private Integer intervalId;

    public Value(BigDecimal value) {
        this.value = value;
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
