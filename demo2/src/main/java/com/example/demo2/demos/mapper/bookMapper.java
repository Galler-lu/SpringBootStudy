package com.example.demo2.demos.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo2.demos.domain.book;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface bookMapper extends BaseMapper<book> {
}
