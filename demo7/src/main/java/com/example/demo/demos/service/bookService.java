package com.example.demo.demos.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.demo.demos.domain.book;

import java.util.List;

public interface bookService extends IService<book> {
    Page<book> selectPage(Integer current, Integer size);
}
