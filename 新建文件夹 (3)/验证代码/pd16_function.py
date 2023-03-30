def pd16hs(account, test_time, a):
    # 0/患者7823
    if account == 13101097823 and test_time == "2020-12-16 00:00:00%2020-12-16 23:59:59":
        if a >= 64670 and a <= 64803:
            a = str(a) + '秒'
        elif a >= 64640 and a < 64670:
            a = str(a) + '前'

    # 1/患者5413
    elif account == 15154365413 and test_time == "2020-12-18 00:00:00%2020-12-18 14:43:33":
        if a >= 38306 and a <= 38360:
            a = str(a) + '秒'
        elif a >= 38306 - 30 and a < 38306:
            a = str(a) + '前'
        elif 39157 <= a <= 39194:
            a = str(a) + '秒'
        elif 39157 - 30 <= a <= 39157:
            a = str(a) + '前'

    # 2/患者3703
    elif account == 15847173073 and test_time == "2020-12-20 00:00:00%2020-12-20 23:59:59":
        if a >= 32312 and a <= 32405:
            a = str(a) + '秒'
        elif a >= 32312 - 30 and a < 32312:
            a = str(a) + '前'
        elif 54503 <= a <= 54549:
            a = str(a) + '秒'
        elif 54503 - 30 <= a <= 54503:
            a = str(a) + '前'
        elif 58058 <= a <= 58094:
            a = str(a) + '秒'
        elif 58058 - 30 <= a <= 58058:
            a = str(a) + '前'

    elif account == 15847173073 and test_time == "2020-12-20 16:00:00%2020-12-20 17:00:00":
        if a >= 959 and a <= 1005:
            a = str(a) + '秒'
        elif a >= 929 and a < 959:
            a = str(a) + '前'
    elif account == 15847173073 and test_time == "2020-12-20 17:00:00%2020-12-20 18:00:00":
        if a >= 914 and a <= 950:
            a = str(a) + '秒'
        elif a >= 884 and a < 914:
            a = str(a) + '前'

    # 3/患者7467
    elif account == 13961327467 and test_time == "2020-12-24 00:00:00%2020-12-24 23:59:59":
        if a >= 76159 and a <= 76367:
            a = str(a) + '秒'
        elif a >= 76129 and a < 76159:
            a = str(a) + '前'
    elif account == 13961327467 and test_time == "2020-12-25 00:00:00%2020-12-25 23:59:59":
        if a >= 72712 and a <= 72833:
            a = str(a) + '秒'
        elif a >= 72712 - 30 and a < 72712:
            a = str(a) + '前'

    # 4/患者1800
    elif account == 15511931800 and test_time == "2020-12-26 00:00:00%2020-12-26 09:36:11":
        if a >= 25095 and a <= 25149:
            a = str(a) + '秒'
        elif a >= 25095-30 and a < 25095:
            a = str(a) + '前'

    # 5/患者1208
    elif account == 15820761208 and test_time == "2020-12-31 00:00:00%2020-12-31 23:59:59":
        if a >= 21722 and a <= 21780:
            a = str(a) + '秒'
        elif a >= 21722-30 and a < 21722:
            a = str(a) + '前'
    elif account == 15820761208 and test_time == "2021-01-03 00:00:00%2021-01-03 23:59:59":
        if a >= 19143 and a <= 19189:
            a = str(a) + '秒'
        elif a >= 19143-30 and a < 19143:
            a = str(a) + '前'

    # 6/患者9768
    elif account == 13796889768 and test_time == "2020-12-30 00:00:00%2020-12-30 23:59:59":
        if a >= 23881 and a <= 23928:
            a = str(a) + '秒'
        elif a >= 23881-30 and a < 23881:
            a = str(a) + '前'
        elif 62383 <= a <= 62427:
            a = str(a) + '秒'
        elif 62383 - 30 <= a <= 62383:
            a = str(a) + '前'
    elif account == 13796889768 and test_time == "2020-12-30 17:30:00%2020-12-30 18:30:00":
        if a >= 2021 and a <= 2065:
            a = str(a) + '秒'
        elif a >= 1991 and a < 2021:
            a = str(a) + '前'

    # 7/患者1909
    elif account == 18732141909 and test_time == "2021-01-10 00:00:00%2021-01-10 23:59:59":
        if a >= 40101 and a <= 40116:
            a = str(a) + '秒'
        elif a >= 40101-30 and a < 40101:
            a = str(a) + '前'
        elif 53503 <= a <= 53594:
            a = str(a) + '秒'
        elif 53503 - 30 <= a <= 53503:
            a = str(a) + '前'
    elif account == 18732141909 and test_time == "2021-01-10 19:00:00%2021-01-10 20:00:00":
        if a >= 3079 - 22 and a <= 3170 - 22:
            a = str(a) + '秒'
        elif a >= 3049 - 22 and a < 3079 - 22:
            a = str(a) + '前'
    elif account == 18732141909 and test_time == "2021-01-11 00:00:00%2021-01-11 23:58:22":
        if a >= 11525 and a <= 11625:
            a = str(a) + '秒'
        elif a >= 11525-30 and a < 11525:
            a = str(a) + '前'

    # 8/患者2558
    elif account == 13930692558 and test_time == "2021-01-24 00:00:00%2021-01-24 23:59:59":
        # if 1744-679 <= a <= 1938-679:
        #     a = str(a) + '秒'
        # elif 1714-679 <= a < 1744-679:
        #     a = str(a) + '前'
        if 310539 - 236265 <= a <= 310588 - 236265:
            a = str(a) + '秒'
        elif 310539 - 236265 - 30 <= a < 310539 - 236265:
            a = str(a) + '前'
        elif 314024 - 236265 <= a <= 314218 - 236265:
            a = str(a) + '秒'
        elif 314024 - 236265 - 30 <= a < 314024 - 236265:
            a = str(a) + '前'
    # elif account == 13930692558 and test_time == "2021-01-24 21:00:00%2021-01-24 22:00:00":
    #     if a>= 2981 and a<=3030:
    #         a=str(a)+'秒'
    #     elif a >= 2951 and a < 2981:
    #         a = str(a) + '前'

    # 9/患者7651
    elif account == 17637907651 and test_time == "2021-02-24 00:00:00%2021-02-24 09:34:09":
        if a >= 29489 and a <= 29561:
            a = str(a) + '秒'
        elif a >= 29459 and a < 29489:
            a = str(a) + '前'

    # 10/患者6067
    elif account == 15558396067 and test_time == "2021-03-12 09:12:08%2021-03-12 20:27:08":
        if a >= 20752 and a <= 20788:
            a = str(a) + '秒'
        elif a >= 20752-30 and a < 20752:
            a = str(a) + '前'

    # 11/患者3569
    elif account == 15805363569 and test_time == "2021-04-22 00:00:00%2021-04-22 01:39:50":
        if a >= 1881 and a <= 1949:
            a = str(a) + '秒'
        elif a >= 1851 and a < 1881:
            a = str(a) + '前'

    # 12/患者3708
    elif account == 18638113708 and test_time =="2021-04-30 00:00:00%2021-04-30 23:59:59":
        if a >= 66795 and a <= 66858:
            a = str(a) + '秒'
        elif a >= 66795-30 and a < 66795:
            a = str(a) + '前'
    elif account == 18638113708 and test_time == "2021-05-01 00:00:00%2021-05-01 23:59:59":
        if a >= 14305 and a <= 14430:
            a = str(a) + '秒'
        elif a >= 14305-30 and a < 14305:
            a = str(a) + '前'
        elif 42573 <= a <= 42686:
            a = str(a) + '秒'
        elif 42573 - 30 <= a < 42573:
            a = str(a) + '前'
    elif account == 18638113708 and test_time == "2021-05-01 19:30:00%2021-05-01 20:30:00":
        if a >= 1637 and a <= 1750:
            a = str(a) + '秒'
        elif a >= 1607 and a < 1637:
            a = str(a) + '前'

    # 13/患者5782
    elif account == 13474285782 and test_time == "2021-05-20 00:00:00%2021-05-20 11:02:08":
        if a >= 27110 and a <= 27120:
            a = str(a) + '秒'
        elif a >= 27110-30 and a < 27110:
            a = str(a) + '前'
        elif 36601 <= a <= 36647:
            a = str(a) + '秒'
        elif 36601 - 30 <= a < 36601:
            a = str(a) + '前'
    elif account == 13474285782 and test_time == "2021-05-20 10:00:00%2021-05-20 11:00:00":
        if a >= 2367 and a <= 2413:
            a = str(a) + '秒'
        elif a >= 2337 and a < 2367:
            a = str(a) + '前'

    # 14/患者7997
    elif account == 15036067997 and test_time == "2021-05-30 00:00:00%2021-05-30 23:59:59":
        if a >= 16718 and a <= 16814:
            a = str(a) + '秒'
        elif a >= 16688 and a < 16718:
            a = str(a) + '前'

    # 15/患者6212
    elif account == 13921976212 and test_time == "2021-10-20 00:00:00%2021-10-20 03:37:20":
        if a >= 12540 and a <= 12634:
            a = str(a) + '秒'
        elif a >= 12540-30 and a < 12540:
            a = str(a) + '前'

    # 16/患者6092
    elif account == 13889886092 and test_time == "2021-10-23 00:00:00%2021-10-23 23:59:59":
        if a >= 61632 and a <= 61725:
            a = str(a) + '秒'
        elif a >= 61602 and a < 61632:
            a = str(a) + '前'

    # 17/患者15975021597 王彦鹏 WYP-46644
    elif account == 15975021597 and test_time == "2021-01-09 00:00:00%2021-01-09 23:59:59":
        if a >= 53771 and a <= 53894:
            a = str(a) + '秒'
        elif a >= 53771 - 30 and a < 53771:
            a = str(a) + '前'

    # 18/患者14784500848 简彦君 JYJ-46985
    elif account == 14784500848 and test_time == "2021-01-20 00:00:00%2021-01-20 23:59:59":
        if a >= 17595 and a <= 17697:
            a = str(a) + '秒'
        elif a >= 17595-30 and a < 17595:
            a = str(a) + '前'
    elif account == 14784500848 and test_time == "2021-01-23 00:00:00%2021-01-23 23:59:59":
        if a >= 234337 - 216043 and a <= 234480 - 216043:
            a = str(a) + '秒'
        elif a >= 234337 - 216043 - 30 and a < 234337 - 216043:
            a = str(a) + '前'

    # 19/患者4727
    elif account == 13838374727 and test_time == "2020-12-20 00:00:00%2020-12-20 23:59:59":
        if a >= 24839 and a <= 24851:
            a = str(a) + '秒'
        elif a >= 24839-30 and a < 24839:
            a = str(a) + '前'
        elif a >= 25161 and a <= 25176:
            a = str(a) + '秒'
        elif a >= 25161-30 and a < 25161:
            a = str(a) + '前'
        elif a >= 36376 and a <= 36390:
            a = str(a) + '秒'
        elif a >= 36376-30 and a < 36376:
            a = str(a) + '前'
        elif a >= 37007 and a <= 37018:
            a = str(a) + '秒'
        elif a >= 37007-30and a < 37007:
            a = str(a) + '前'
        elif a >= 49080 and a <= 49132:
            a = str(a) + '秒'
        elif a >= 49080-30and a < 49080:
            a = str(a) + '前'
    elif account == 13838374727 and test_time == "2020-12-20 07:05:00%2020-12-20 08:05:00":
        if a >= 286 - 17 and a <= 301 - 17:
            a = str(a) + '秒'
        elif a >= 256 - 17 and a < 286 - 17:
            a = str(a) + '前'
    elif account == 13838374727 and test_time == "2020-12-20 11:30:00%2020-12-20 12:30:00":
        if a >= 202 and a <= 216:
            a = str(a) + '秒'
        elif a >= 172 and a < 202:
            a = str(a) + '前'
        elif a >= 833 and a <= 844:
            a = str(a) + '秒'
        elif a >= 803 and a < 833:
            a = str(a) + '前'
    elif account == 13838374727 and test_time == "2020-12-20 15:00:00%2020-12-20 16:00:00":
        if a >= 306 and a <= 358:
            a = str(a) + '秒'
        elif a >= 276 and a < 306:
            a = str(a) + '前'

    return (a)
