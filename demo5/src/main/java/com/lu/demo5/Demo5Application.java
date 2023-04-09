package com.lu.demo5;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(basePackages = "com.lu.demo5.dao")
public class Demo5Application {

    public static void main(String[] args) {
        SpringApplication.run(Demo5Application.class, args);
    }

}
