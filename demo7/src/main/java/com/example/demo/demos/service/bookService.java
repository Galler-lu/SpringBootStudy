package com.example.demo.demos.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.demo.demos.domain.book;

import java.util.List;

public interface bookService extends IService<book> {
    List<book> selectPage(Integer current,Integer size);
}
