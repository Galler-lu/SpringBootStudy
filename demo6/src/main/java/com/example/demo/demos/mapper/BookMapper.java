package com.example.demo.demos.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.demos.domain.Books;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface BookMapper extends BaseMapper<Books> {
}
