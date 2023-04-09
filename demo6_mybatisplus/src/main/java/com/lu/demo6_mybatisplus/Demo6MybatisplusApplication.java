package com.lu.demo6_mybatisplus;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(basePackages = "com.lu.demo6_mybatisplus.mapper")
public class Demo6MybatisplusApplication {

    public static void main(String[] args) {
        SpringApplication.run(Demo6MybatisplusApplication.class, args);
    }

}
