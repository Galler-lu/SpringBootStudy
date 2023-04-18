package com.example.demo.demos.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo.demos.domain.Books;
import com.example.demo.demos.mapper.BookMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class BookServiceImp extends ServiceImpl<BookMapper, Books> implements BookService{
    @Autowired
    private BookMapper bookMapper;
    @Override
    public List<Books> pageSelect(Integer current, Integer size) {
        Page<Books> booksPage = new Page<>(current, size);
        Page<Books> page = bookMapper.selectPage(booksPage, null);
        return page.getRecords();
    }
}
