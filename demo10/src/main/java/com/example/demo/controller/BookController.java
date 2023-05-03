package com.example.demo.controller;

import com.example.demo.demos.Book;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class BookController {
    @GetMapping("/getByID")
    public String getByID(){
        return "springboot";
    }
    @GetMapping("/books")
    public Book books(){
        Book book = new Book(1001, "测试", "必读", "仅做测试");
        return book;
    }
}
