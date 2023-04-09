package com.lu.demo6_mybatisplus.service.Imp;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.lu.demo6_mybatisplus.domain.Books;
import com.lu.demo6_mybatisplus.mapper.bookMapper;
import com.lu.demo6_mybatisplus.service.bookService;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Service;

@Service
public class bookServiceImp extends ServiceImpl<bookMapper, Books> implements bookService {
}
