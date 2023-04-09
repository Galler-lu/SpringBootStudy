package com.lu.ssmp.service.serviceImp;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.lu.ssmp.dao.booksDao;
import com.lu.ssmp.domain.books;
import com.lu.ssmp.service.bookService;

//通过继承ServiceImpl来避免实现bookService接口时要实现的方法，
// 传参时注意泛型，第一个参数为BaseMapper的子类，第二个参数为要操作的对象
public class booksServiceImp extends ServiceImpl<booksDao, books> implements bookService {
}
