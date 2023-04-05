package com.java.lu.demo3.demos.dao.daoImp;

import com.java.lu.demo3.demos.dao.bookDao;
import org.springframework.stereotype.Repository;

@Repository
public class bookDaoImp implements bookDao {
    @Override
    public String test1() {
        return "test1.....";
    }

    @Override
    public String test2() {
        return "test2....";
    }
}
