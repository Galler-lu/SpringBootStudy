package com.lu.demo6_mybatisplus;

import com.lu.demo6_mybatisplus.domain.Books;
import com.lu.demo6_mybatisplus.service.bookService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.ArrayList;

@SpringBootTest
class Demo6MybatisplusApplicationTests {
    @Autowired
    private com.lu.demo6_mybatisplus.service.bookService bookService;
    @Test
    void contextLoads() {
        //查询总记录数
        long count = bookService.count();
        System.out.println("总记录数："+count);
    }
    //批量插入
    @Test
    public void insertBatchTest(){
        ArrayList<Books> books = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Books book = new Books();
            book.setBookCounts(i);
            book.setBookName("不要进厂"+i);
            book.setDetail("狗都不进厂的人数"+i);
            books.add(book);
        }
        boolean saveBatch = bookService.saveBatch(books);
        if (saveBatch==true){
            System.out.println("插入成功");
        }else {
            System.out.println("插入失败");
        }
    }


}
