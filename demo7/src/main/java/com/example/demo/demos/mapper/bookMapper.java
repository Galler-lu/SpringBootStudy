package com.example.demo.demos.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.demos.domain.book;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

//@Mapper
public interface bookMapper extends BaseMapper<book> {
}
