package com.example.demo.demos.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/test")
public class helloSpringBoot {
    @GetMapping("/")
    public String hello(){
        System.out.println("hello,springboot1...");
        return "hello,springboot1...";
    }
}
