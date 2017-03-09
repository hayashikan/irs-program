# -*- coding: utf-8 -*-
"""
Project: MAM Integrated Reporting System
Author: LIN, Han (Jo)
"""

import os, sys, json, copy
import pandas as pd
import numpy as np
import datetime, math
import calendar

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


class Integrated_Reporting_System(object):

    class ColWidthError(Exception):
        pass

    class FileNotFoundError(Exception):
        pass

    def __init__(self, json_spec_path):
        json_file = open(json_spec_path, 'r')
        self.spec = json.load(json_file, encoding='utf-8')
        self.spec = byteify(self.spec)
        self.folders = self.file_validate()
        self.curdf = dict()
        self.full_agglist = None
        self.csv_config = {
            'encoding': 'utf-8-sig',
            'index': False
        }


    def file_validate(self):
        root = self.spec['Directory']['Open']
        if self.spec['Frequency']=='Monthly' or self.spec['Frequency']=='Weekday-weekend (Monthly data)':
            w = 'M'
        if self.spec['Frequency']=='Daily' or self.spec['Frequency']=='Daily Average' or self.spec['Frequency']=='Weekday-weekend (Daily data)':
            w = 'D'
        files = [f for f in os.listdir(root) 
            if os.path.isfile(os.path.join(root, f)) and f.startswith(w)]
        folders = set([f.split('.')[0] for f in files])
        for folder in folders:
            for sfx in self.spec['parse'].keys():
                if folder + '.' + sfx not in files:
                    raise FileNotFoundError('%s.%s is not in input directory!')
        folders = sorted(folders, reverse=False)
        return folders


    def parse_fwf(self, idx, suffix):
        # need revision
        f = os.path.join(self.spec['Directory']['Open'], '%s.%s' % (self.folders[idx], suffix))
        widths = self.spec['parse'][suffix]['widths']
        names = self.spec['parse'][suffix]['names']
        usecols = self.spec['parse'][suffix]['usecols']
        if len(widths)!=len(names):
            raise self.ColWidthError('%s lenth of column list and name list do not match' % suffix)
        converters = dict()
        for name in names:
            converters[name] = str
        df = pd.read_fwf(f, widths=widths, header=None, names=names,
                         converters=converters, encoding='utf-8',
                         usecols=usecols)
        return df


    def parse_folder(self, idx):
        for suffix in self.spec['parse'].keys():
            self.curdf[suffix] = self.parse_fwf(idx, suffix)

    def charencode(self, x):
        try:
            return x.decode('utf-8').encode('utf-8')
        except:
            return "Python Cannot Identify"


    def pre_usage_report_process(self):
        if self.spec['Object']=='App':
            if self.spec['ReportType']['Type']=='Usage Report' or \
            self.spec['ReportType']['Type']=='Usage By Target Report':
                # Transform
                self.curdf['DEM']['Individual_Weighting'] = self.curdf['DEM']['Individual_Weighting'].apply(
                    lambda x: float(x))
                self.curdf['APPSWD']['Start_Time'] = self.curdf['APPSWD']['Start_Time'].apply(lambda 
                    x:datetime.datetime.strptime(x, '%H%M%S'))
                self.curdf['APPSWD']['End_Time'] = self.curdf['APPSWD']['End_Time'].apply(lambda 
                    x:datetime.datetime.strptime(x, '%H%M%S'))
                self.curdf['APPSWD']['ts'] = self.curdf['APPSWD']['End_Time'] - \
                    self.curdf['APPSWD']['Start_Time']
                self.curdf['APPSWD']['ts'] = self.curdf['APPSWD']['ts'].apply(
                    lambda x: x / np.timedelta64(1, 's') + 1)

                # Join Weighting to APPSWD
                self.curdf['APPSWD'] = pd.merge(left=self.curdf['APPSWD'],
                    right=self.curdf['DEM'][['Individual_ID', 'Individual_Weighting']],
                    left_on='Individual_ID', right_on='Individual_ID', how='left')
                self.curdf['APPSWD']['tswgt'] = self.curdf['APPSWD']['ts'] * self.curdf['APPSWD']['Individual_Weighting']
                self.curdf['APPSWD'].drop(['Start_Time', 'End_Time', 'ts'], axis=1, inplace=True)
        if self.spec['Object']=='Web':
            if self.spec['ReportType']['Type']=='Usage Report' or \
            self.spec['ReportType']['Type']=='Usage By Target Report':
                self.curdf['DEM']['Individual_Weighting'] = self.curdf['DEM']['Individual_Weighting'].apply(
                    lambda x: float(x))
                self.curdf['WEBSWD'] = pd.merge(left=self.curdf['WEBSWD'],
                    right=self.curdf['DEM'][['Individual_ID', 'Individual_Weighting']],
                    left_on='Individual_ID', right_on='Individual_ID', how='left')


    def confirm_metric(self, slabeltime, elabeltime, stime, etime, weighting):
        if self.spec['Metrics'] == ['Reach (000)']:
            return self.confirm_reach(slabeltime, elabeltime, stime, etime, weighting)
        if self.spec['Metrics'] == ['Total Minutes (000)']:
            return self.confirm_time(slabeltime, elabeltime, stime, etime, weighting)
        raise KeyError('Too many metrics! Please choose only ONE metric for day part report')


    def confirm_time(self, slabeltime, elabeltime, stime, etime, weighting):
        if stime <= elabeltime and etime >= slabeltime:
            stime = stime.to_datetime()
            etime = etime.to_datetime()
            start_time = max(stime, slabeltime)
            end_time = min(etime, elabeltime)
            return (end_time - start_time).total_seconds() * weighting
        else:
            return 0


    def confirm_reach(self, slabeltime, elabeltime, stime, etime, weighting):
        if stime <= elabeltime and etime >= slabeltime:
            return weighting
        else:
            return 0

    def pre_daypart_report_process(self):
        self.curdf['DEM']['Individual_Weighting'] = self.curdf['DEM']['Individual_Weighting'].apply(
            lambda x: float(x))
        self.curdf['APPSWD']['Start_Time'] = self.curdf['APPSWD']['Start_Time'].apply(lambda 
            x:datetime.datetime.strptime(x, '%H%M%S'))
        self.curdf['APPSWD']['End_Time'] = self.curdf['APPSWD']['End_Time'].apply(lambda 
            x:datetime.datetime.strptime(x, '%H%M%S'))
        self.curdf['APPSWD'] = pd.merge(left=self.curdf['APPSWD'],
            right=self.curdf['DEM'][['Individual_ID', 'Individual_Weighting']],
            left_on='Individual_ID', right_on='Individual_ID', how='left')

        start_time = self.spec['ReportType']['StartTime']
        start_time = datetime.datetime.strptime(start_time, '%H%M%S')
        end_time = self.spec['ReportType']['EndTime']
        end_time = datetime.datetime.strptime(end_time, '%H%M%S')

        # Time Slots Transform
        cur_stime = start_time
        interval = self.spec['ReportType']['Interval']
        td = datetime.timedelta(minutes=interval)
        timeslots = []
        while cur_stime < end_time:
            cur_etime = cur_stime + td - datetime.timedelta(seconds=1)
            cur_etime = min(cur_etime, end_time)
            stime_label = datetime.datetime.strftime(cur_stime, '%H:%M:%S')
            etime_label = datetime.datetime.strftime(cur_etime, '%H:%M:%S')
            cur_label = stime_label + ' - ' + etime_label
            timeslots.append(cur_label)
            if self.spec['Metrics'] == ['Reach (000)']:
                self.curdf['APPSWD'][cur_label] = self.curdf['APPSWD'].apply(
                    lambda row: self.confirm_reach(cur_stime, cur_etime, 
                        row['Start_Time'], row['End_Time'], row['Individual_Weighting']), axis=1)
            if self.spec['Metrics'] == ['Total Minutes (000)']:
                self.curdf['APPSWD'][cur_label] = self.curdf['APPSWD'].apply(
                    lambda row: self.confirm_time(cur_stime, cur_etime, 
                        row['Start_Time'], row['End_Time'], row['Individual_Weighting']), axis=1)
            cur_stime += td
            print cur_label
        self.spec['ReportType']['timeslots'] = timeslots
        self.period_metric = timeslots


    def pre_aggregate_process(self):
        if self.spec['Object']=='App':
            # Aggregate Level Configure
            if self.spec['UnitOfAnalysis']['Type']=='Category':
                self.curdf['APPSWD'] = pd.merge(left=self.curdf['APPSWD'],
                    right=self.curdf['APPCLA'][['Application', 'Category']],
                    left_on='Application', right_on='Application',
                    how='left')
                self.curdf['APPSWD'].drop(['Application'], axis=1, inplace=True)
                self.agglist = self.curdf['APPCATMST']
            if self.spec['UnitOfAnalysis']['Type']=='Subcategory':
                self.curdf['APPCLA']['Subcategory'] = self.curdf['APPCLA']['Category'] + self.curdf['APPCLA']['Subcategory']
                self.curdf['APPSWD'] = pd.merge(left=self.curdf['APPSWD'],
                    right=self.curdf['APPCLA'][['Application', 'Subcategory']],
                    left_on='Application', right_on='Application',
                    how='left')
                self.curdf['APPSWD'].drop(['Application'], axis=1, inplace=True)
                self.curdf['APPSUBCATMST']['Category'] = self.curdf['APPSUBCATMST']['Subcategory'].apply(lambda x:x[:8])
                self.curdf['APPSUBCATMST'] = pd.merge(left=self.curdf['APPSUBCATMST'],
                    right=self.curdf['APPCATMST'],
                    left_on='Category', right_on='Category', how='left')
                self.agglist = self.curdf['APPSUBCATMST'][['Subcategory', 'Subcategory_Name', 'Category_Name']]
            if self.spec['UnitOfAnalysis']['Type']=='Application':
                self.curdf['APPCLA']['Subcategory'] = \
                    self.curdf['APPCLA']['Category'] + \
                    self.curdf['APPCLA']['Subcategory']
                self.curdf['APPCLA'] = pd.merge(left=self.curdf['APPCLA'],
                    right=self.curdf['APPCATMST'], left_on='Category',
                    right_on='Category', how='left')
                self.curdf['APPCLA'] = pd.merge(left=self.curdf['APPCLA'],
                    right=self.curdf['APPSUBCATMST'], left_on='Subcategory',
                    right_on='Subcategory', how='left')
                self.curdf['APPLIST'] = pd.merge(left=self.curdf['APPLIST'],
                    right=self.curdf['APPCLA'], left_on='Application', right_on=
                    'Application', how='left')
                self.agglist = self.curdf['APPLIST'][['Application', 
                'Application_Name', 'Category_Name', 'Subcategory_Name']]
                self.curdf['APPLIST'] = None
                self.curdf['APPCLA'] = None
                self.curdf['APPCATMST'] = None
                self.curdf['APPSUBCATMST'] = None
            if self.spec['UnitOfAnalysis']['Type']=='Total':
                self.curdf['APPSWD'][self.spec['UnitOfAnalysis']['AggCol']] = '00000000'
                self.agglist = pd.DataFrame(
                    data={self.spec['UnitOfAnalysis']['AggCol']: ['00000000'],
                    self.spec['UnitOfAnalysis']['AggColName']: ['Total']})
            if self.spec['UnitOfAnalysis']['Type']=='Application Group':
                groupdf = pd.DataFrame()
                groupid = 0
                groupid_list = []
                groupname_list = []
                for group in self.spec['UnitOfAnalysis']['GroupSpec']:
                    group_name = group['GroupName']
                    group_apps = set(group['GroupItems'])
                    curgroupdf = self.curdf['APPSWD'][self.curdf['APPSWD']['Application'].isin(group_apps)]
                    curgroupdf[self.spec['UnitOfAnalysis']['AggCol']] = str(groupid)
                    groupid_list.append(str(groupid))
                    groupname_list.append(group_name)
                    groupdf = groupdf.append(curgroupdf)
                    groupid += 1
                self.curdf['APPSWD'] = groupdf
                self.agglist = pd.DataFrame(
                    data={self.spec['UnitOfAnalysis']['AggCol']:groupid_list,
                    self.spec['UnitOfAnalysis']['AggColName']: groupname_list})
        if self.spec['Object']=='Web':
            if self.spec['UnitOfAnalysis']['Type']=='Category':
                self.curdf['WEBSWD'] = pd.merge(left=self.curdf['WEBSWD'],
                    right=self.curdf['WEBCLA'][['Website', 'Category']],
                    left_on='Website', right_on='Website',
                    how='left')
                self.curdf['WEBSWD'].drop(['Website'], axis=1, inplace=True)
                self.agglist = self.curdf['WEBCATMST']
            if self.spec['UnitOfAnalysis']['Type']=='Subcategory':
                self.curdf['WEBCLA']['Subcategory'] = self.curdf['WEBCLA']['Category'] + self.curdf['WEBCLA']['Subcategory']
                self.curdf['WEBSWD'] = pd.merge(left=self.curdf['WEBSWD'],
                    right=self.curdf['WEBCLA'][['Website', 'Subcategory']],
                    left_on='Website', right_on='Website',
                    how='left')
                self.curdf['WEBSWD'].drop(['Website'], axis=1, inplace=True)
                self.curdf['WEBSUBCATMST']['Category'] = self.curdf['WEBSUBCATMST']['Subcategory'].apply(lambda x:x[:8])
                self.curdf['WEBSUBCATMST'] = pd.merge(left=self.curdf['WEBSUBCATMST'],
                    right=self.curdf['WEBCATMST'],
                    left_on='Category', right_on='Category', how='left')
                self.agglist = self.curdf['WEBSUBCATMST'][['Subcategory', 'Subcategory_Name', 'Category_Name']]
            if self.spec['UnitOfAnalysis']['Type']=='Website':
                self.curdf['WEBCLA']['Subcategory'] = \
                    self.curdf['WEBCLA']['Category'] + \
                    self.curdf['WEBCLA']['Subcategory']
                self.curdf['WEBCLA'] = pd.merge(left=self.curdf['WEBCLA'],
                    right=self.curdf['WEBCATMST'], left_on='Category',
                    right_on='Category', how='left')
                self.curdf['WEBCLA'] = pd.merge(left=self.curdf['WEBCLA'],
                    right=self.curdf['WEBSUBCATMST'], left_on='Subcategory',
                    right_on='Subcategory', how='left')
                self.curdf['WEBLIST'] = pd.merge(left=self.curdf['WEBLIST'],
                    right=self.curdf['WEBCLA'], left_on='Website', right_on=
                    'Website', how='left')
                self.agglist = self.curdf['WEBLIST'][['Website', 
                'Website_Name', 'Category_Name', 'Subcategory_Name']]
                self.curdf['WEBLIST'] = None
                self.curdf['WEBCLA'] = None
                self.curdf['WEBCATMST'] = None
                self.curdf['WEBSUBCATMST'] = None
            if self.spec['UnitOfAnalysis']['Type']=='Total':
                self.curdf['WEBSWD'][self.spec['UnitOfAnalysis']['AggCol']] = '00000000'
                self.agglist = pd.DataFrame(
                    data={self.spec['UnitOfAnalysis']['AggCol']: ['00000000'],
                    self.spec['UnitOfAnalysis']['AggColName']: ['Total']})
            if self.spec['UnitOfAnalysis']['Type']=='Website Group':
                groupdf = pd.DataFrame()
                groupid = 0
                groupid_list = []
                groupname_list = []
                for group in self.spec['UnitOfAnalysis']['GroupSpec']:
                    group_name = group['GroupName']
                    group_apps = set(group['GroupItems'])
                    curgroupdf = self.curdf['WEBSWD'][self.curdf['WEBSWD']['Website'].isin(group_apps)]
                    curgroupdf[self.spec['UnitOfAnalysis']['AggCol']] = str(groupid)
                    groupid_list.append(str(groupid))
                    groupname_list.append(group_name)
                    groupdf = groupdf.append(curgroupdf)
                    groupid += 1
                self.curdf['WEBSWD'] = groupdf
                self.agglist = pd.DataFrame(
                    data={self.spec['UnitOfAnalysis']['AggCol']:groupid_list,
                    self.spec['UnitOfAnalysis']['AggColName']: groupname_list})
        for col in self.agglist.columns:
            self.agglist[col] = self.agglist[col].apply(self.charencode)


    def standard_app_report(self, groupdf, groupname, universe):
        data = {}
        aggcol = self.spec['UnitOfAnalysis']['AggCol']

        reach = groupdf.groupby([aggcol, 'Individual_ID']).agg({
            'Individual_Weighting': np.min,
            aggcol: np.min,
        })
        reach = reach.groupby(aggcol).agg({
            'Individual_Weighting': np.sum,
        })

        if 'Reach (000)' in self.spec['Metrics']:
            data['Reach (000)'] = reach['Individual_Weighting']

        # if 'Installed Reach (000)'  in self.spec['Metrics']:
        #     data['Installed Reach (000)'] = reach['Individual_Weighting']

        # if 'Uninstalled Reach (000)' in self.spec['Metrics']:
        #     data['Uninstalled Reach (000)'] = reach['Individual_Weighting']
            
        if 'Reach (%)' in self.spec['Metrics']:
            reach_percent = reach['Individual_Weighting'] / universe
            data['Reach (%)'] = reach_percent
            
        if 'Total Minutes (000)' in self.spec['Metrics'] or \
        'Time Per User (in minutes)' in self.spec['Metrics']:
            total_minutes = groupdf.groupby(aggcol).agg({
                'tswgt': np.sum
            })
            total_minutes = total_minutes / 60
            
        if 'Total Minutes (000)' in self.spec['Metrics']:
            data['Total Minutes (000)'] = total_minutes['tswgt']
            
        if 'Time Per User (in minutes)' in self.spec['Metrics']:
            time_per_user = total_minutes['tswgt'] / reach['Individual_Weighting']
            data['Time Per User (in minutes)'] = time_per_user
            
        if 'Total Sessions (000)' in self.spec['Metrics'] or \
        'Sessions Per User' in self.spec['Metrics']:
            total_session = groupdf.groupby(aggcol).agg({
                'Individual_Weighting': np.sum
                })
        if 'Total Sessions (000)' in self.spec['Metrics']:
            data['Total Sessions (000)'] = total_session['Individual_Weighting']
            
        if 'Sessions Per User' in self.spec['Metrics']:
            session_per_user = total_session['Individual_Weighting'] / reach['Individual_Weighting']
            data['Sessions Per User'] = session_per_user
            
        df = pd.DataFrame(data=data, index=reach.index)
        df = df[self.spec['Metrics']]
        rename_cols = {col: groupname+': '+col for col in df.columns}
        df.rename(index=str, columns=rename_cols, inplace=True)
        # tups = [(groupname, col) for col in df.columns]
        # df.columns = pd.MultiIndex.from_tuples(tups, 
        #     names=['Group Name','Metric'])
        return df


    def standard_web_report(self, groupdf, groupname, universe):
        data = {}
        aggcol = self.spec['UnitOfAnalysis']['AggCol']

        reach = groupdf.groupby([aggcol, 'Individual_ID']).agg({
            'Individual_Weighting': np.min,
            aggcol: np.min,
        })
        reach = reach.groupby(aggcol).agg({
            'Individual_Weighting': np.sum,
        })

        if 'Reach (000)' in self.spec['Metrics']:
            data['Reach (000)'] = reach['Individual_Weighting']
            
        if 'Reach (%)' in self.spec['Metrics']:
            reach_percent = reach['Individual_Weighting'] / universe
            data['Reach (%)'] = reach_percent
            
        if 'Total Page Views (000)' in self.spec['Metrics'] or \
        'Page Views Per User' in self.spec['Metrics']:
            total_session = groupdf.groupby(aggcol).agg({
                'Individual_Weighting': np.sum
                })
        if 'Total Page Views (000)' in self.spec['Metrics']:
            data['Total Page Views (000)'] = total_session['Individual_Weighting']
            
        if 'Page Views Per User' in self.spec['Metrics']:
            session_per_user = total_session['Individual_Weighting'] / reach['Individual_Weighting']
            data['Page Views Per User'] = session_per_user
            
        df = pd.DataFrame(data=data, index=reach.index)
        df = df[self.spec['Metrics']]
        rename_cols = {col: groupname+': '+col for col in df.columns}
        df.rename(index=str, columns=rename_cols, inplace=True)
        # tups = [(groupname, col) for col in df.columns]
        # df.columns = pd.MultiIndex.from_tuples(tups, 
        #     names=['Group Name','Metric'])
        return df


    def __daypart_app_report_reach(self, groupdf, groupname, universe):
        aggcol = self.spec['UnitOfAnalysis']['AggCol']
        agg_dict = {label:np.max for label in self.spec['ReportType']['timeslots']}
        agg_dict[aggcol] = np.min
        groupdf = groupdf.groupby([aggcol, 'Individual_ID']).agg(agg_dict)
        agg_dict = {label:np.sum for label in self.spec['ReportType']['timeslots']}
        groupdf = groupdf.groupby(aggcol).agg(agg_dict)
        groupdf = groupdf[self.spec['ReportType']['timeslots']]
        return groupdf


    def __daypart_app_report_total_timespent(self, groupdf, groupname, universe):
        aggcol = self.spec['UnitOfAnalysis']['AggCol']
        agg_dict = {label:np.sum for label in self.spec['ReportType']['timeslots']}
        groupdf = groupdf.groupby(aggcol).agg(agg_dict)
        groupdf = groupdf / 60
        groupdf = groupdf[self.spec['ReportType']['timeslots']]
        return groupdf


    def daypart_app_report(self, groupdf, groupname, universe):
        if self.spec['Metrics'] == ['Reach (000)']:
            return self.__daypart_app_report_reach(groupdf, groupname, universe)
        if self.spec['Metrics'] == ['Total Minutes (000)']:
            return self.__daypart_app_report_total_timespent(groupdf, groupname, universe)
        raise KeyError('Too many metrics!')


    def generate_single_daypart_app_report(self, idx, parse=True, pre_process=True):
        if parse:
            self.parse_folder(idx)
        if pre_process:
            self.pre_aggregate_process()
        self.pre_daypart_report_process()

        groupdf = self.curdf['APPSWD']
        groupname = 'All'
        universe = self.curdf['DEM']['Individual_Weighting'].sum()
        daypart_report = self.daypart_app_report(groupdf, groupname, universe)
        daypart_report = pd.merge(left=self.agglist, right=daypart_report,
            left_on=self.spec['UnitOfAnalysis']['AggCol'], right_index=True, how='right')
        return daypart_report


    def generate_single_standard_app_report(self, idx, parse=True, pre_process=True):
        if parse:
            self.parse_folder(idx)
        if pre_process:
            self.pre_aggregate_process()
        self.pre_usage_report_process()
        
        groupdf = self.curdf['APPSWD']
        groupname = 'All'
        universe = self.curdf['DEM']['Individual_Weighting'].sum()
        std_report = self.standard_app_report(groupdf, groupname, universe)
        std_report = pd.merge(left=self.agglist, right=std_report,
            left_on=self.spec['UnitOfAnalysis']['AggCol'], 
            right_index=True, how='right')
        return std_report


    def generate_single_installation_app_report(self, idx, parse=True, pre_process=True):
        pass


    def generate_single_by_target_installation_app_report(self, idx, parse=True, pre_process=True):
        pass


    def generate_single_data_usage_report(self, idx, parse=True, pre_process=True):
        pass


    def generate_single_by_target_data_usage_report(self, idx, parse=True, pre_process=True):
        pass


    def generate_single_standard_web_report(self, idx, parse=True, pre_process=True):
        if parse:
            self.parse_folder(idx)
        if pre_process:
            self.pre_aggregate_process()
        self.pre_usage_report_process()
        
        groupdf = self.curdf['WEBSWD']
        groupname = 'All'
        universe = self.curdf['DEM']['Individual_Weighting'].sum()
        std_report = self.standard_web_report(groupdf, groupname, universe)
        std_report = pd.merge(left=self.agglist, right=std_report,
            left_on=self.spec['UnitOfAnalysis']['AggCol'], 
            right_index=True, how='right')
        return std_report


    def generate_single_by_target_app_report(self, idx, parse=True, pre_process=True):
        main_report = self.generate_single_standard_app_report(idx, parse, pre_process)
        for tg_group in self.spec['ReportType']['TargetGroup']:
            groupname = tg_group['GroupName']
            groupquery = tg_group['Query']
            groupuser = set(self.curdf['DEM'].query(groupquery)['Individual_ID'].unique())
            universe = self.curdf['DEM'].query(groupquery)['Individual_Weighting'].sum()
            groupdf = self.curdf['APPSWD'][self.curdf['APPSWD']['Individual_ID'].isin(groupuser)]
            group_report = self.standard_app_report(groupdf, groupname, universe)
            main_report = pd.merge(left=main_report, right=group_report,
                left_on=self.spec['UnitOfAnalysis']['AggCol'], right_index=True, how='left')
        main_report = main_report.replace(np.nan, 0)
        return main_report


    def generate_single_by_target_web_report(self, idx, parse=True, pre_process=True):
        main_report = self.generate_single_standard_web_report(idx, parse, pre_process)
        for tg_group in self.spec['ReportType']['TargetGroup']:
            groupname = tg_group['GroupName']
            groupquery = tg_group['Query']
            groupuser = set(self.curdf['DEM'].query(groupquery)['Individual_ID'].unique())
            universe = self.curdf['DEM'].query(groupquery)['Individual_Weighting'].sum()
            groupdf = self.curdf['WEBSWD'][self.curdf['WEBSWD']['Individual_ID'].isin(groupuser)]
            group_report = self.standard_web_report(groupdf, groupname, universe)
            main_report = pd.merge(left=main_report, right=group_report,
                left_on=self.spec['UnitOfAnalysis']['AggCol'], right_index=True, how='left')
        main_report = main_report.replace(np.nan, 0)
        return main_report


    def generate_single_by_target_daypart_app_report(self, idx, parse=True, pre_process=True):
        main_report = self.generate_single_daypart_app_report(idx, parse, pre_process)
        daypart_groupdfs = [main_report]
        daypart_groupnames = ['All']
        for tg_group in self.spec['ReportType']['TargetGroup']:
            groupname = tg_group['GroupName']
            groupquery = tg_group['Query']
            groupuser = set(self.curdf['DEM'].query(groupquery)['Individual_ID'].unique())
            universe = self.curdf['DEM'].query(groupquery)['Individual_Weighting'].sum()
            groupdf = self.curdf['APPSWD'][self.curdf['APPSWD']['Individual_ID'].isin(groupuser)]
            group_report = self.daypart_app_report(groupdf, groupname, universe)
            group_report = pd.merge(left=self.agglist, right=group_report,
                left_on=self.spec['UnitOfAnalysis']['AggCol'], right_index=True, how='right')
            daypart_groupdfs.append(group_report)
            daypart_groupnames.append(groupname)
        main_report = pd.concat(daypart_groupdfs, keys=daypart_groupnames)
        main_report.reset_index(level=0, inplace=True)
        main_report.rename(columns={main_report.columns[0]: 'Target Group'}, inplace=True)
        return main_report


    def monthly_weekday_weekend_analysis(self, idx):
        parse = False
        pre_process = False
        self.parse_folder(idx)
        self.pre_aggregate_process()
        self.__valid_metrics()

        weekday_df = pd.DataFrame()
        weekend_df = pd.DataFrame()
        num_weekdays = 0
        num_weekends = 0
        year = int(self.folders[idx][1:5])
        month = int(self.folders[idx][-2:])
        days_of_month = calendar.monthrange(year, month)[1]

        if self.spec['Object']=='App':
            monthly_df = self.curdf['APPSWD']
        if self.spec['Object']=='Web':
            monthly_df = self.curdf['WEBSWD']

        # separate days
        for day in range(1, days_of_month+1):
            current_date = str(year) + str(month).zfill(2) + str(day).zfill(2)
            print 'processing: %d / %d days of %d %d' % (day, days_of_month, year, month)
            day_number = datetime.date(year, month, day).weekday()
            if self.spec['Object']=='App':
                self.curdf['APPSWD'] = monthly_df[monthly_df['Date']==current_date]
                if self.spec['ReportType']['Type']=='Usage Report':
                    day_df = self.generate_single_standard_app_report(idx, parse, pre_process)
                if self.spec['ReportType']['Type']=='Usage By Target Report':
                    day_df = self.generate_single_by_target_app_report(idx, parse, pre_process)
                if self.spec['ReportType']['Type']=='Usage Day Part Report':
                    day_df = self.generate_single_daypart_app_report(idx, parse, pre_process)
                if self.spec['ReportType']['Type']=='Usage Day Part By Target Report':
                    day_df = self.generate_single_by_target_daypart_app_report(idx, parse, pre_process)
            if self.spec['Object']=='Web':
                self.curdf['WEBSWD'] = monthly_df[monthly_df['Date']==current_date]
                if self.spec['ReportType']['Type']=='Usage Report':
                    day_df = self.generate_single_standard_web_report(idx, parse, pre_process)
                if self.spec['ReportType']['Type']=='Usage By Target Report':
                    day_df = self.generate_single_by_target_web_report(idx, parse, pre_process)
            if day_number <= 4:
                weekday_df = weekday_df.append(day_df)
                num_weekdays += 1
            else:
                weekend_df = weekend_df.append(day_df)
                num_weekends += 1

        # average method
        self.full_agglist = self.agglist
        self.agglist_cols = list(self.agglist.columns)
        self.agglist_cols.remove(self.spec['UnitOfAnalysis']['AggCol'])
        weekday_df.drop(self.agglist_cols, axis=1, inplace=True)
        weekday_df = self.average_report(weekday_df, num_weekdays)
        weekend_df.drop(self.agglist_cols, axis=1, inplace=True)
        weekend_df = self.average_report(weekend_df, num_weekends)

        report = pd.concat([weekday_df, weekend_df], keys=['Weekday', 'Weekend'])
        report.reset_index(level=0, inplace=True)
        report.rename(columns={report.columns[0]: 'Day of Week'}, inplace=True)
        return report


    def daily_weekday_weekend_analysis(self, idx, cmd, weekday_df=None, weekend_df=None, num_weekdays=0,
        num_weekends=0, df=None):

        if cmd == 'init':
            weekday_df = pd.DataFrame()
            weekend_df = pd.DataFrame()
            return (weekday_df, weekend_df, num_weekdays, num_weekends)

        if cmd == 'append':
            if not isinstance(df, pd.DataFrame):
                raise KeyError('')
            d = self.folders[idx]
            year = int(d[1:5])
            month = int(d[5:7])
            day = int(d[7:])
            day_number = datetime.date(year, month, day).weekday()
            if day_number <= 4:
                weekday_df = weekday_df.append(df)
                return (weekday_df, weekend_df, num_weekdays+1, num_weekends)
            if day_number > 4:
                weekend_df = weekend_df.append(df)
                return (weekday_df, weekend_df, num_weekdays, num_weekends+1)

        if cmd == 'average':
            if len(weekday_df) > 0:
                weekday_df = self.average_report(weekday_df, num_weekdays)
            if len(weekend_df) > 0:
                weekend_df = self.average_report(weekend_df, num_weekends)
            report = pd.concat([weekday_df, weekend_df], keys=['Weekday', 'Weekend'])
            report.reset_index(level=0, inplace=True)
            report.rename(columns={report.columns[0]: 'Day of Week'}, inplace=True)
            return report


    def generate_single_report(self, idx):
        if self.spec['Frequency'] != 'Weekday-weekend (Monthly data)':
            if self.spec['Object']=='App':
                if self.spec['ReportType']['Type']=='Usage Report':
                    return self.generate_single_standard_app_report(idx)
                if self.spec['ReportType']['Type']=='Usage By Target Report':
                    return self.generate_single_by_target_app_report(idx)
                if self.spec['ReportType']['Type']=='Usage Day Part Report':
                    return self.generate_single_daypart_app_report(idx)
                if self.spec['ReportType']['Type']=='Usage Day Part By Target Report':
                    return self.generate_single_by_target_daypart_app_report(idx)
                # if self.spec['ReportType']['Type']=='Installation Report':
                #     return self.generate_single_installation_app_report(idx)
                # if self.spec['ReportType']['Type']=='Installation By Target Report':
                #     return self.generate_single_by_target_installation_app_report(idx)
                # if self.spec['ReportType']['Type']=='Data Usage Report':
                #     return self.generate_single_data_usage_report(idx)
                # if self.spec['ReportType']['Type']=='Data Usage By Target Report':
                #     return self.generate_single_by_target_data_usage_report(idx)
            if self.spec['Object']=='Web':
                if self.spec['ReportType']['Type']=='Usage Report':
                    return self.generate_single_standard_web_report(idx)
                if self.spec['ReportType']['Type']=='Usage By Target Report':
                    return self.generate_single_by_target_web_report(idx)
        if self.spec['Frequency'] == 'Weekday-weekend (Monthly data)':
            print 'Weekday-weekend (Monthly data)'
            return self.monthly_weekday_weekend_analysis(idx)



    def __update_new_agglist_value(self, new_aggname, old, new):
        if new_aggname==0:
            return old
        return new


    def update_full_agglist(self):
        if not isinstance(self.full_agglist, pd.DataFrame):
            self.agglist_cols = [col for col in self.agglist.columns \
                if col!=self.spec['UnitOfAnalysis']['AggCol']]
            return self.agglist

        full_agglist = self.full_agglist
        agglist = self.agglist
        rename_cols = {s: s+'_' for s in self.agglist_cols}
        agglist = agglist.rename(index=str, columns=rename_cols)
        full_agglist = pd.merge(
            left=full_agglist, right=agglist,
            left_on=self.spec['UnitOfAnalysis']['AggCol'],
            right_on=self.spec['UnitOfAnalysis']['AggCol'],
            how='outer')
        full_agglist = full_agglist.replace(np.nan, 0)
        for col in self.agglist_cols:
            full_agglist[col] = full_agglist.apply(lambda row: 
                self.__update_new_agglist_value(
                    row[self.spec['UnitOfAnalysis']['AggColName']+'_'],
                    row[col], row[col+'_']), axis=1)
        drop_cols = [col+'_' for col in self.agglist_cols]
        full_agglist.drop(drop_cols, axis=1, inplace=True)
        return full_agglist


    def daily_average_report(self, full_report, num_days=None):
        # average method & clean format
        cols = list(full_report.columns)
        metric_cols = copy.deepcopy(cols)
        metric_cols.remove(self.spec['UnitOfAnalysis']['AggCol'])
        avg_aggcol = self.spec['UnitOfAnalysis']['AggCol']
        if 'Target Group' in metric_cols:
            metric_cols.remove('Target Group')
            avg_aggcol = ['Target Group', self.spec['UnitOfAnalysis']['AggCol']]
        agg_dict = {col: np.sum for col in metric_cols}

        # aggregate
        daily_average_report = full_report.groupby(avg_aggcol).agg(agg_dict)
        daily_average_report = daily_average_report[metric_cols]

        # weighted average for 3 metrics
        if num_days==None:
            num_days = len(self.folders)
        for col in metric_cols:
            daily_average_report[col] = daily_average_report[col] / num_days
        for col in metric_cols:
            groupname = col.rsplit(':', 1)[0]
            if 'Time Per User (in minutes)' in col:
                daily_average_report[col] = daily_average_report[groupname + ': Total Minutes (000)']\
                    / daily_average_report[groupname + ': Reach (000)']
                continue
            if 'Sessions Per User' in col:
                daily_average_report[col] = daily_average_report[groupname + ': Total Sessions (000)']\
                    / daily_average_report[groupname + ': Reach (000)']
                continue
            if 'Page Views Per User' in col:
                daily_average_report[col] = daily_average_report[groupname + ': Total Page Views (000)']\
                    / daily_average_report[groupname + ': Reach (000)']
                continue

        # delete unwanted columns
        drop_cols = []
        for col in metric_cols:
            if self.spec['ReportType']['Type']!='Usage Day Part Report' and self.spec['ReportType']['Type']!='Usage Day Part By Target Report':
                m = col.rsplit(':', 1)[1]
            else:
                m = col
            if m.strip() not in self.period_metric:
                drop_cols.append(col)
        daily_average_report.drop(drop_cols, axis=1, inplace=True)

        # join updated agglist
        if 'Target Group' in cols:
            daily_average_report.reset_index(inplace=True)
            daily_average_report.set_index(self.spec['UnitOfAnalysis']['AggCol'], inplace=True)
        daily_average_report = pd.merge(left=self.full_agglist, right=daily_average_report,
            left_on=self.spec['UnitOfAnalysis']['AggCol'],
            right_index=True, how='right')

        # formatting
        if 'Target Group' in cols:
            cols = list(daily_average_report.columns)
            cols.remove('Target Group')
            cols = ['Target Group'] + cols
            daily_average_report = daily_average_report[cols]

        daily_average_report = daily_average_report.replace(np.nan, 0)
        return daily_average_report


    def average_report(self, full_report, num_days=None):
        return self.daily_average_report(full_report, num_days)


    def __valid_metrics(self):
        self.period_metric = copy.deepcopy(self.spec['Metrics'])
        if 'Time Per User (in minutes)' in self.period_metric:
            if 'Total Minutes (000)' not in self.period_metric:
                self.spec['Metrics'].append('Total Minutes (000)')
            if 'Reach (000)' not in self.spec['Metrics']:
                self.spec['Metrics'].append('Reach (000)')
        if 'Sessions Per User' in self.period_metric:
            if 'Total Sessions (000)' not in self.period_metric:
                self.spec['Metrics'].append('Total Sessions (000)')
            if 'Reach (000)' not in self.spec['Metrics']:
                self.spec['Metrics'].append('Reach (000)')
        if 'Page Views Per User' in self.period_metric:
            if 'Total Page Views (000)' not in self.period_metric:
                self.spec['Metrics'].append('Total Page Views (000)')
            if 'Reach (000)' not in self.spec['Metrics']:
                self.spec['Metrics'].append('Reach (000)')
        print self.spec['Metrics'], self.period_metric


    def generate_report(self):
        save_single_report = False
        period_average = False
        period_weekly_average = False
        if self.spec['Frequency'] == 'Daily' or self.spec['Frequency'] == 'Monthly' or self.spec['Frequency'] == 'Weekday-weekend (Monthly data)':
            save_single_report = True
        if self.spec['Frequency'] == 'Daily Average':
            period_average = True
            full_report = pd.DataFrame()
            self.__valid_metrics()
        if self.spec['Frequency'] == 'Weekday-weekend (Daily data)':
            period_weekly_average = True
            self.__valid_metrics()
            weekday_df, weekend_df, num_weekdays, num_weekends = \
                self.daily_weekday_weekend_analysis(0, cmd='init', num_weekdays=0, num_weekends=0)

        for idx in range(len(self.folders)):
            print 'processing: %d / %d folders' % (idx+1, len(self.folders))
            single_report = self.generate_single_report(idx)
            if save_single_report:
                report_name = '%s %s %s %s.csv'
                file_name = os.path.join(self.spec['Directory']['Save'], report_name \
                        % (self.spec['Frequency'], self.spec['Object'], self.spec['ReportType']['Type'], \
                            self.folders[idx][1:]))
                single_report.to_csv(file_name, **self.csv_config)
            if period_average or period_weekly_average:
                self.full_agglist = self.update_full_agglist()
                single_report.drop(self.agglist_cols, inplace=True, axis=1)
            if period_average:
                full_report = full_report.append(single_report)
            if period_weekly_average:
                weekday_df, weekend_df, num_weekdays, num_weekends = \
                    self.daily_weekday_weekend_analysis(idx, cmd='append', weekday_df=weekday_df,
                        weekend_df=weekend_df, num_weekdays=num_weekdays, num_weekends=num_weekends, df=single_report)

        if period_average or period_weekly_average:
            if period_average:
                daily_average_report = self.average_report(full_report)
            if period_weekly_average:
                daily_average_report = self.daily_weekday_weekend_analysis(0, cmd='average', 
                    weekday_df=weekday_df, weekend_df=weekend_df, num_weekdays=num_weekdays, num_weekends=num_weekends)
            report_name = '%s %s %s %s - %s.csv'
            file_name = os.path.join(self.spec['Directory']['Save'], report_name \
                % (self.spec['Frequency'], self.spec['Object'], self.spec['ReportType']['Type'], \
                    self.folders[0][1:], self.folders[-1][1:]))
            daily_average_report.to_csv(file_name, **self.csv_config)

