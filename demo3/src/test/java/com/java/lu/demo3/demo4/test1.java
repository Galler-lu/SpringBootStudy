package com.java.lu.demo3.demo4;

import com.java.lu.demo3.demos.dao.bookDao;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class test1 {
    @Autowired
    private com.java.lu.demo3.demos.dao.bookDao bookDao;
    @Test
    public void test(){
        System.out.println(bookDao.test1());
    }
}
