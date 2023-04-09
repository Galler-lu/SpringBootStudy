package com.lu.demo4.demos.dao;

import com.lu.demo4.demos.domain.books;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Repository;

import java.util.List;

@Mapper
public interface booksDao {
    //根据bookID查询书籍
    @Select("select *from ssmbuild.books where bookID=#{bookID}")
    books getBookByID(Integer bookID);
    //获取全部书籍
    @Select("select *from ssmbuild.books")
    List<books> getAllBooks();
}
