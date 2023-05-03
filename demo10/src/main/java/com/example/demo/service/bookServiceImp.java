package com.example.demo.service;

import com.example.demo.dao.bookDao;
import com.example.demo.demos.Book;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class bookServiceImp implements bookService{
    @Autowired
    private com.example.demo.dao.bookDao bookDao;
    @Override
    public void insert(Book book) {
       bookDao.insert(book);

    }
}
