package com.example.demo2;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.demo2.demos.domain.book;
import com.example.demo2.demos.mapper.bookMapper;
import com.example.demo2.demos.service.bookService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
public class test1 {
    @Autowired
    private com.example.demo2.demos.service.bookService bookService;
    @Autowired
    private com.example.demo2.demos.mapper.bookMapper bookMapper;
    @Test
    public void test1(){
        //根据bookID查询
        book book = bookMapper.selectById(1);
        System.out.println(book);
    }
    @Test
    public void test2(){
        //根据条件查询
        Integer bookCounts=5;
        QueryWrapper<book> bookQueryWrapper = new QueryWrapper<>();
        bookQueryWrapper.gt("bookID",1010).like("detail","不进厂")
                        .between(bookCounts>4,"bookCounts",bookCounts,9);
        List<book> books = bookMapper.selectList(bookQueryWrapper);
        books.forEach(System.out::println);
    }
    @Test
    public void test3(){
        //分页查询
        Page<book> bookPage = new Page<>(0, 3);
        Page<book> bookPage1 = bookMapper.selectPage(bookPage, null);
        bookPage1.getRecords().forEach(System.out::println);
    }
}
