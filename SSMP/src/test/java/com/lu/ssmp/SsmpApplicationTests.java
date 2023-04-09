package com.lu.ssmp;

import com.lu.ssmp.dao.booksDao;
import com.lu.ssmp.domain.books;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
class SsmpApplicationTests {

    @Autowired
    private com.lu.ssmp.dao.booksDao booksDao;

    @Test
    void testGetById() {
        books books = booksDao.selectById(1);
        System.out.println(books);
    }

    @Test
    void testGetAll() {
        List<books> books = booksDao.selectList(null);
        for (com.lu.ssmp.domain.books book : books) {
            System.out.println(book);
        }

    }
    @Test
    void testSave() {
        books books = new books(1001, "狗都不进厂", 100, "不要进场，不要进场");
        int insert = booksDao.insert(books);
        if (insert==1){
            System.out.println("插入成功："+insert);
        }else {
            System.out.println("插入失败");
        }
    }
    @Test
    void testUpdate() {
        booksDao.updateById(new books(1001, "不进厂", 100, "不要进场，不要进场"));
    }
    @Test
    void testDelete() {
        booksDao.deleteById(new books(1001, "不进厂", 100, "不要进场，不要进场"));
    }
    @Test
    void testPage() {

    }
}
