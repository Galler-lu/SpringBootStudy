package com.example.demo.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.demos.Book;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Component;
@Component
public interface bookDao extends BaseMapper<Book> {
}
