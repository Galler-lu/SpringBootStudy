package com.example.demo;

import com.example.demo.demos.controller.helloSpringBoot;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

@SpringBootApplication
public class Demo1Application {

    public static void main(String[] args) {
//        引导类得到的是一个容器
        ConfigurableApplicationContext context = SpringApplication.run(Demo1Application.class, args);
        helloSpringBoot bean = context.getBean(helloSpringBoot.class);
        System.out.println("bean=======>"+bean);
        student bean1 = context.getBean(student.class);
        System.out.println("bean1======>"+bean1);
    }

}
