package com.example.demo.demos.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@TableName("book")
public class Books {
    @TableId(value = "bookID",type = IdType.AUTO)
    private Integer bookID;
    private String bookName;
    private Integer bookCounts;
    private String detail;
}
