/*==============================================================*/
/* DBMS name:      Sybase SQL Anywhere 11                       */
/* Created on:     2023/5/17 18:28:59                           */
/*==============================================================*/


if exists(select 1 from sys.sysforeignkey where role='FK_COUNSELL_MANAGE_STUDENT') then
    alter table counsellor
       delete foreign key FK_COUNSELL_MANAGE_STUDENT
end if;

if exists(select 1 from sys.sysforeignkey where role='FK_EXCUSE_EXAMINE_COUNSELL') then
    alter table excuse
       delete foreign key FK_EXCUSE_EXAMINE_COUNSELL
end if;

if exists(select 1 from sys.sysforeignkey where role='FK_EXCUSE_WRITE2_STUDENT') then
    alter table excuse
       delete foreign key FK_EXCUSE_WRITE2_STUDENT
end if;

if exists(select 1 from sys.sysforeignkey where role='FK_STUDENT_WRITE_EXCUSE') then
    alter table student
       delete foreign key FK_STUDENT_WRITE_EXCUSE
end if;

if exists(
   select 1 from sys.systable 
   where table_name='counsellor'
     and table_type in ('BASE', 'GBL TEMP')
) then
    drop table counsellor
end if;

if exists(
   select 1 from sys.systable 
   where table_name='excuse'
     and table_type in ('BASE', 'GBL TEMP')
) then
    drop table excuse
end if;

if exists(
   select 1 from sys.systable 
   where table_name='student'
     and table_type in ('BASE', 'GBL TEMP')
) then
    drop table student
end if;

/*==============================================================*/
/* Table: counsellor                                            */
/*==============================================================*/
create table counsellor 
(
   "Employee No"        char(20)                       not null,
   "student ID"         char(20)                       null,
   name                 char(20)                       null,
   "Manage Class"       char(20)                       null,
   constraint PK_COUNSELLOR primary key ("Employee No")
);

/*==============================================================*/
/* Table: excuse                                                */
/*==============================================================*/
create table excuse 
(
   information          char(20)                       not null,
   "Employee No"        char(20)                       not null,
   "student ID"         char(20)                       null,
   "Reason for Leave"   char(50)                       null,
   "Leave time"         char(20)                       null,
   "Back to school time" char(20)                       null,
   constraint PK_EXCUSE primary key (information)
);

/*==============================================================*/
/* Table: student                                               */
/*==============================================================*/
create table student 
(
   "student ID"         char(20)                       not null,
   information          char(20)                       null,
   name                 char(20)                       null,
   class                char(20)                       null,
   constraint PK_STUDENT primary key ("student ID")
);

alter table counsellor
   add constraint FK_COUNSELL_MANAGE_STUDENT foreign key ("student ID")
      references student ("student ID")
      on update restrict
      on delete restrict;

alter table excuse
   add constraint FK_EXCUSE_EXAMINE_COUNSELL foreign key ("Employee No")
      references counsellor ("Employee No")
      on update restrict
      on delete restrict;

alter table excuse
   add constraint FK_EXCUSE_WRITE2_STUDENT foreign key ("student ID")
      references student ("student ID")
      on update restrict
      on delete restrict;

alter table student
   add constraint FK_STUDENT_WRITE_EXCUSE foreign key (information)
      references excuse (information)
      on update restrict
      on delete restrict;

