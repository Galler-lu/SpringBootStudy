package com.example.demo.demos.controller;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.example.demo.demos.domain.Books;
import com.example.demo.demos.service.BookService;
import com.example.demo.demos.utils.R;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/books")
public class BookController {
    @Autowired
    private BookService bookService;

    //根据id查询数据
    @GetMapping("{id}")
    public R getById(@PathVariable("id") Integer id) {
        return new R(true,bookService.getById(id));
    }

    //查询全部数据
    @GetMapping
    public R getAll() {
        return new R(true,bookService.list());
    }

    //修改数据
    @PutMapping
    public R updateBooks(@RequestBody Books books) {
//        return bookService.update(books,null);//此处为修改全部数据
        return new R(true,bookService.updateById(books));
    }
    //添加数据
    @PostMapping
    public R saveBooks(@RequestBody Books books){
        return new R(true,bookService.save(books));
    }
    //根据id删除数据
    @DeleteMapping("/{id}")
    public R deleteBooks(@PathVariable Integer id){
        return new R(true,bookService.removeById(id));
    }
    //分页查询
    @GetMapping("/{current}/{size}")
    public R selectPage(@PathVariable("current") Integer current,@PathVariable("size") Integer size){
        return new R(true,bookService.pageSelect(current,size));
    }
}
