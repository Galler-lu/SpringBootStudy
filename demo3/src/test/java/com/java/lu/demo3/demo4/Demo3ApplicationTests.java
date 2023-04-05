package com.java.lu.demo3.demo4;

import com.java.lu.demo3.Demo3Application;
import com.java.lu.demo3.demos.dao.bookDao;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;

//注意：这种加了@SpringBootTest注解的类要在Demo3Application这种类的同一包及其子包下，
// 若不符合上述情况则需要在SpringBootTest注解里配置class属性，声明启动类，
// 或者增加一个注解@ContextConfiguration,并配置其中的class属性
@SpringBootTest//(classes = Demo3Application.class)
//@ContextConfiguration(classes = Demo3Application.class)
class Demo3ApplicationTests {

    @Autowired
    private com.java.lu.demo3.demos.dao.bookDao bookDao;
    @Test
    void contextLoads() {
        System.out.println(bookDao.test1());
    }

}
