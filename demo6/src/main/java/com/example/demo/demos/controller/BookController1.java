package com.example.demo.demos.controller;

import com.example.demo.demos.domain.Books;
import com.example.demo.demos.service.BookService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/books1")
public class BookController1 {
    @Autowired
    private BookService bookService;

    //根据id查询数据
    @GetMapping("{id}")
    public Books getById(@PathVariable("id") Integer id) {
        return bookService.getById(id);
    }

    //查询全部数据
    @GetMapping
    public List<Books> getAll() {
        return bookService.list();
    }

    //修改数据
    @PutMapping
    public Boolean updateBooks(@RequestBody Books books) {
//        return bookService.update(books,null);//此处为修改全部数据
        return bookService.updateById(books);
    }
    //添加数据
    @PostMapping
    public Boolean saveBooks(@RequestBody Books books){
        return bookService.save(books);
    }
    //根据id删除数据
    @DeleteMapping("/{id}")
    public Boolean deleteBooks(@PathVariable Integer id){
        return bookService.removeById(id);
    }
    //分页查询
    @GetMapping("/{current}/{size}")
    public List<Books> selectPage(@PathVariable("current") Integer current,@PathVariable("size") Integer size){
        return bookService.pageSelect(current,size);
    }
}
