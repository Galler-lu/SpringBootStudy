package com.example.demo.demos.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.demo.demos.domain.Books;

import java.util.List;

public interface BookService  extends IService<Books> {
    public List<Books> pageSelect(Integer current,Integer size);
}
