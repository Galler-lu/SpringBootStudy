package com.example.demo;

import com.example.demo.demos.Book;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.Rollback;
import org.springframework.transaction.annotation.Transactional;

@SpringBootTest
@Transactional
@Rollback(value = false)
public class Demo10Test1 {
    @Autowired
    private com.example.demo.service.bookService bookService;
    @Test
    void test4(){
        Book books = new Book();
        books.setDescription("仅测试使用2");
        books.setName("测试用例2");
        books.setType("测试2");
        books.setId(1011);
        bookService.insert(books);
    }
}
