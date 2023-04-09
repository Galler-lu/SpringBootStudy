package com.lu.demo4;

import com.lu.demo4.demos.dao.booksDao;
import com.lu.demo4.demos.domain.books;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
class Demo4ApplicationTests {
    @Autowired
    private com.lu.demo4.demos.dao.booksDao booksDao;

    @Test
    void contextLoads() {
        books bookByID = booksDao.getBookByID(1);
        System.out.println(bookByID);
    }
    @Test
    public void test1(){
        List<books> allBooks = booksDao.getAllBooks();
        for (books allBook : allBooks) {
            System.out.println(allBook);
        }
    }

}
