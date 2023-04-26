package com.example.demo.demos.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.demo.demos.domain.book;
import com.example.demo.demos.service.bookService;
import com.example.demo.demos.utils.R;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import javax.swing.text.rtf.RTFEditorKit;
import java.util.List;

@RestController
@RequestMapping("book")
public class bookController {
    @Autowired
    private com.example.demo.demos.service.bookService bookService;
    @GetMapping("{id}")
    public R getById(@PathVariable Integer id){
        book book = bookService.getById(id);
        return new R(true,book);
    }
    @GetMapping
    public R getAll(){
        List<book> list = bookService.list();
        return new R(true,list);
    }
    @GetMapping("/{current}/{size}")
    public R getPage(@PathVariable("current") Integer current,@PathVariable("size") Integer size){
        Page<book> bookPage = bookService.selectPage(current, size);
        return new R(true,bookPage);
    }
    //修改数据
    @PutMapping
    public R updateBooks(@RequestBody book book) {
//        return bookService.update(books,null);//此处为修改全部数据
        return new R(true,bookService.updateById(book));
    }
    //添加数据
    @PostMapping
    public R saveBooks(@RequestBody book book){
        return new R(true,bookService.save(book));
    }
    //根据id删除数据
    @DeleteMapping("/{id}")
    public R deleteBooks(@PathVariable Integer id){
        return new R(true,bookService.removeById(id));
    }
}
