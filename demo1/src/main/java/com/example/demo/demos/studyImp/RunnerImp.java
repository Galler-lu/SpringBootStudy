package com.example.demo.demos.studyImp;

import com.example.demo.demos.ExtendClass.Car;
import com.example.demo.demos.ExtendClass.Person;
import com.example.demo.demos.study01.CastMoney;
import com.example.demo.demos.study01.Runner;

public class RunnerImp extends Car implements Runner, CastMoney {

    @Override
    public void run() {
        System.out.println("快跑");
    }

    @Override
    public void car_method() {

        System.out.println("RunnerImp的car_method方法");
    }

    @Override
    public void stop() {
        System.out.println("停止");
    }

    @Override
    public void start() {
        System.out.println("开始");
    }

    @Override
    public void light() {
        System.out.println("车开灯");
    }

    @Override
    public void cast() {
        System.out.println("RunnerImp的cast方法");
    }

}
