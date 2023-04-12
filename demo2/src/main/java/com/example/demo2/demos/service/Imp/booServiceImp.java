package com.example.demo2.demos.service.Imp;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo2.demos.domain.book;
import com.example.demo2.demos.mapper.bookMapper;
import com.example.demo2.demos.service.bookService;
import org.springframework.stereotype.Service;

@Service
public class booServiceImp extends ServiceImpl<bookMapper, book> implements bookService {
}
