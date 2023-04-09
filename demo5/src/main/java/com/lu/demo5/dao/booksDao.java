package com.lu.demo5.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.lu.demo5.domain.books;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

//@Mapper
@Repository
public interface booksDao extends BaseMapper<books> {
}
