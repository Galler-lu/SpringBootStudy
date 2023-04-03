package com.java.lu.demo3.demos.controller;
//package com.java.lu.demo3.demos.controller;
//
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class hello {

   @RequestMapping("/hello1")
    public String hello(){
        System.out.println("hello word");
        return "hello,world";
    }
}

