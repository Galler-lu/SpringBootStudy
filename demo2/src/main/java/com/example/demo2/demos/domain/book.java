package com.example.demo2.demos.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class book {
    @TableId(value = "bookID",type = IdType.AUTO)
    private Integer bookID;
    private String bookName;
    private Integer bookCounts;
    private String detail;
}
