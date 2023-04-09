package com.lu.demo6_mybatisplus.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.lu.demo6_mybatisplus.domain.Books;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Repository;

@Repository
public interface bookMapper extends BaseMapper<Books> {
}
