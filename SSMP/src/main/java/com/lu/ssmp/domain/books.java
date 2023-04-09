package com.lu.ssmp.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class books {
    private Integer bookID;
    private String bookName;
    private Integer bookCounts;
    private String detail;
}
