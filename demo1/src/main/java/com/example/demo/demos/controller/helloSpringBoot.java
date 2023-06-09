package com.example.demo.demos.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/test")
public class helloSpringBoot {
    @GetMapping("/test1")
    public String hello(){
        System.out.println("hello,springboot1...");
        return "hello,springboot1...";
    }
    @GetMapping("/test2")
    public String test2(){
        System.out.println("hello,springboot2...");
        System.out.println("master....1");
        System.out.println("master1....1");
        System.out.println("push....1");
        System.out.println("pull....1");
        return "hello,springboot2...";
    }
    @RequestMapping(value = "/users/{id}",method = RequestMethod.GET)
    @ResponseBody
    public String getById(@PathVariable("id") Integer id){
        System.out.println("users getById==>"+id);
        return "{'module':'user getByID'}";
    }
}
