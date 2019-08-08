def trimModelList(files):
    '''
        get_xml_files(files)

        Based on Paul Durack's trimModelList. The function returns a list
        of xml files for a given (untrimmed list of files). This function will 
        return one xml per model / realization prioritizing the latest data and 
        then published data. Inputs include:

    '''    

    # get all models / realizations
    models = []
    realizations = []
    for f in files:
        m = f.split('/')[-1].split('.')[4]
        r = f.split('/')[-1].split('.')[5]
        models.append(m)
        realizations.append(r)
    models = list(set(models))
    realizations = list(set(realizations))
    base = '/'.join(f.split('/')[0:-1])

    tfiles = []
    # loop over models and realizations to find
    # simulations with multiple versions
    for m in models:
        for r in realizations:
            xstring = base + '/*.' + m + '.*.' + r
            l = findInList(xstring, files)

            if len(l) == 0:
                continue;
            elif len(l) == 1:
                fOut = l[0].replace('//','/')
                tfiles.append(fOut)
            else:
                ver = []
                # prioritize integer version numbers
                # the v + integers
                # then take latest data by date
                # then 'latest'
                cdates = []
                for f in l:
                    # get creation dates
                    fh = cdms2.open(f)
                    cdate = fh.creation_date
                    # most dates are of form: 2012-02-13T00:40:33Z
                    # some are: Thu Aug 11 22:49:09 EST 2011 - just make 20110101
                    if cdate[0].isalpha():
                        cdate = int(cdate.split(' ')[-1] + '0101')
                    else:
                        cdate = int(cdate.split('T')[0].replace('-',''))
                    cdates.append(cdate)
                    fh.close()
                # check for max creation date
                cdates = np.array(cdates)
                dmax = np.where(cdates.max() == cdates)[0]
                if len(dmax) == 1:
                    ind = dmax[0]
                    fOut = l[ind].replace('//','/')
                    tfiles.append(fOut)                    
                else:
                    l = np.take(l,dmax)
                    for f in l:
                        v = f.split('/')[-1].split('.')[10]
                        if v == 'latest':
                            v = 0
                        elif v.find('v') > 0:
                            v = v*1000000000
                        else:
                            v = int(v.replace('v',''))
                            if v < 10:
                                v = v*100000000
                        ver.append(v)
                    ver = np.array(ver)
                    ind=np.where(ver.max() == ver)[0][0]
                    fOut = l[ind].replace('//','/')
                    tfiles.append(fOut)
                # uncomment to print file choices
                # l[ind] = l[ind] + ' *'
                # for f in l:
                #     print(f)
                # print(' ')
    return tfiles