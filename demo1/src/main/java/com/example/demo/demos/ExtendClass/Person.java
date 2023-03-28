package com.example.demo.demos.ExtendClass;

public class Person {
    private Character sex;
    private String name;

    @Override
    public String toString() {
        return "Person{" +
                "sex=" + sex +
                ", name='" + name + '\'' +
                '}';
    }

    public Character getSex() {
        return sex;
    }

    public void setSex(Character sex) {
        this.sex = sex;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Person() {
    }

    public Person(Character sex, String name) {
        this.sex = sex;
        this.name = name;
    }
}
