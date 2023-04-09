package com.lu.demo5;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.conditions.query.LambdaQueryChainWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.lu.demo5.dao.booksDao;
import com.lu.demo5.domain.books;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
class Demo5ApplicationTests {

    @Autowired
    private com.lu.demo5.dao.booksDao booksDao;
    @Test
    void contextLoads() {
        books books = booksDao.selectById(1);
        System.out.println(books);

//        System.out.println("----------------------------------------");
//        List<com.lu.demo5.domain.books> books1 = booksDao.selectList(null);
//        for (com.lu.demo5.domain.books books2 : books1) {
//            System.out.println(books2);
//        }
    }
    @Test
    public void test1(){
        Page<books> booksPage = booksDao.selectPage(new Page<books>(1, 3), new LambdaQueryWrapper<books>().gt(books::getBookID, 1010));
        booksPage.getRecords().forEach(System.out::println);
    }

}
