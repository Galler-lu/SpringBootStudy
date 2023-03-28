package com.example.demo.demos.study01;

public interface CastMoney {
    int price=10;
    void cast();
    //java8:接口中允许有默认方法
    default void test1(){
        System.out.println("CastMoney的test方法");
    }
//    java8:接口中允许有静态方法
    static void lifeLive(){
        System.out.println("CastMoney的lifeLive方法");
    }
}
