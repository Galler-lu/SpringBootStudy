package com.example.demo.demos.service.Imp;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo.demos.domain.book;
import com.example.demo.demos.mapper.bookMapper;
import com.example.demo.demos.service.bookService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class bookServiceImp extends ServiceImpl<bookMapper, book> implements bookService {

    @Autowired
    private bookMapper bookMapper;

    @Override
    public Page<book> selectPage(Integer current, Integer size) {
        Page<book> bookPage = new Page<>(current, size);
        Page<book> page = bookMapper.selectPage(bookPage, null);
        if (current>bookPage.getPages()){
            page = bookMapper.selectPage(new Page<book>(bookPage.getPages(),size),null);
        }

//        List<book> records = page.getRecords();
        return page;
    }
}
