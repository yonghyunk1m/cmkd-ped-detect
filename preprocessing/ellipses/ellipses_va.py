from matplotlib.patches import Ellipse

def ellipse_va_5242023(station):
    
    ellipses_data = {}

    if station == "Cadell":
        # 1-meter buffer
        ellipse1_1m = Ellipse((770, 645), 70, 40, angle=0) 
        ellipse2_1m = Ellipse((905, 520), 70, 20, angle=0)
        ellipse3_1m = Ellipse((1230, 740), 80, 40, angle=4)
        ellipse4_1m = Ellipse((710, 500), 65, 20, angle=0)
        ellipse5_1m = Ellipse((905, 490), 60, 20, angle=0)
        ellipse6_1m = Ellipse((1590, 680), 90, 35, angle=4)
        ellipses_data["1m"] = [ellipse1_1m, ellipse2_1m, ellipse3_1m, ellipse4_1m, ellipse5_1m, ellipse6_1m]

        # 3-meter buffer
        ellipse1_3m = Ellipse((770, 645), 200, 90, angle=0)
        ellipse2_3m = Ellipse((905, 520), 140, 40, angle=0)
        ellipse3_3m = Ellipse((1230, 740), 270, 140, angle=4)
        ellipse4_3m = Ellipse((710, 500), 110, 40, angle=0)
        ellipse5_3m = Ellipse((905, 490), 120, 40, angle=0)
        ellipse6_3m = Ellipse((1590, 680), 250, 90, angle=6)
        ellipses_data["3m"] = [ellipse1_3m, ellipse2_3m, ellipse3_3m, ellipse4_3m, ellipse5_3m, ellipse6_3m]

        # 6-meter buffer
        ellipse1_6m = Ellipse((770, 670), 400, 180,angle=0)
        ellipse2_6m = Ellipse((905, 525), 280, 80, angle=0)
        ellipse3_6m = Ellipse((1230, 790), 500, 280, angle=4)
        ellipse4_6m = Ellipse((710, 500), 220, 80, angle=0) 
        ellipse5_6m = Ellipse((905, 495), 260, 80, angle=0)
        ellipse6_6m = Ellipse((1590, 700), 500, 200, angle=6)
        ellipses_data["6m"] = [ellipse1_6m, ellipse2_6m, ellipse3_6m, ellipse4_6m, ellipse5_6m, ellipse6_6m]

        # 9-meter buffer
        ellipse1_9m = Ellipse((770, 680), 660, 280, angle=0)
        ellipse2_9m = Ellipse((905, 525), 420, 115, angle=0)
        ellipse3_9m = Ellipse((1230, 815), 880, 400, angle=4)
        ellipse4_9m = Ellipse((710, 500), 330, 105, angle=0)
        ellipse5_9m = Ellipse((905, 495), 380, 100, angle=0)
        ellipse6_9m = Ellipse((1590, 730), 750, 380, angle=6)
        ellipses_data["9m"] = [ellipse1_9m, ellipse2_9m, ellipse3_9m, ellipse4_9m, ellipse5_9m, ellipse6_9m]
        
    elif station == "TechwayA":
        ellipses_data["1m"] = [Ellipse((980, 540), 180, 50, angle=2), Ellipse((660, 490), 65, 20, angle=0), Ellipse((555, 480), 50, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((980, 550), 380, 100, angle=-2), Ellipse((660, 495), 240, 40, angle=-2), Ellipse((555, 480), 170, 20, angle=-2)]
        ellipses_data["6m"] = [Ellipse((1000, 580), 760, 210, angle=-1), Ellipse((660, 500), 460, 60,angle=-2), Ellipse((555, 480), 360, 28, angle=-2)]
        ellipses_data["9m"] = [Ellipse((1000, 620), 1180, 330, angle=-2), Ellipse((660, 520), 650, 80, angle=-2), Ellipse((560, 485), 450, 30, angle=-2)]
 
    elif station == "TechwayB":
        ellipses_data["1m"] = [Ellipse((1250, 530), 80, 20, angle=2), Ellipse((990, 490), 40, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1250, 530), 260, 30, angle=2), Ellipse((990, 490), 175, 25, angle=0)]
        ellipses_data["6m"] = [Ellipse((1250, 530), 520, 70, angle=2), Ellipse((990, 495), 350, 30, angle=0)]
        ellipses_data["9m"] = [Ellipse((1250, 570), 760, 150, angle=2), Ellipse((990, 495), 500, 45, angle=2)]
               
    elif station == "TechwayC":
        ellipses_data["1m"] = [Ellipse((1200, 505), 180, 40, angle=3.8), Ellipse((910, 445), 80, 25, angle=2)]
        ellipses_data["3m"] = [Ellipse((1200, 510), 400, 110, angle=3.5), Ellipse((905, 450), 200, 30, angle=1.5)]
        ellipses_data["6m"] = [Ellipse((1200, 530), 650, 155, angle=3), Ellipse((905, 450), 300, 50, angle=1)]
        ellipses_data["9m"] = [Ellipse((1200, 540), 900, 220, angle=2), Ellipse((910, 450), 500, 60, angle=1)]
        
    elif station == "TechwayD":
        ellipses_data["1m"] = [Ellipse((370, 650), 65, 30, angle=0), Ellipse((605, 495), 35, 20, angle=1)]
        ellipses_data["3m"] = [Ellipse((370, 650), 210, 53, angle=1), Ellipse((605, 495), 150, 40, angle=1)]
        ellipses_data["6m"] = [Ellipse((380, 650), 480, 125, angle=1), Ellipse((620, 495), 280, 70, angle=1)]
        ellipses_data["9m"] = [Ellipse((415, 650), 620, 180, angle=1), Ellipse((630, 495), 400, 105, angle=1)]
        
    return(ellipses_data)

def ellipse_va_6012023(station):
    
    ellipses_data = {}
    
    if station == "Cadell":
        # 1-meter buffer
        ellipse1_1m = Ellipse((760, 470), 70, 40, angle=0) 
        ellipse2_1m = Ellipse((900, 350), 70, 20, angle=0) 
        ellipse3_1m = Ellipse((1215, 580), 80, 40, angle=4) 
        ellipse4_1m = Ellipse((730, 300), 65, 20, angle=0) 
        ellipse5_1m = Ellipse((905, 315), 60, 20, angle=0) 
        ellipse6_1m = Ellipse((1570, 540), 90, 35, angle=4) 
        ellipses_data["1m"] = [ellipse1_1m, ellipse2_1m, ellipse3_1m, ellipse4_1m, ellipse5_1m, ellipse6_1m]

        # 3-meter buffer
        ellipse1_3m = Ellipse((760, 470), 200, 90, angle=0)
        ellipse2_3m = Ellipse((900, 350), 140, 40, angle=0)
        ellipse3_3m = Ellipse((1215, 580), 270, 140, angle=4)
        ellipse4_3m = Ellipse((730, 300), 110, 40, angle=0)
        ellipse5_3m = Ellipse((905, 315), 120, 40, angle=0)
        ellipse6_3m = Ellipse((1570, 540), 250, 90, angle=6)
        ellipses_data["3m"] = [ellipse1_3m, ellipse2_3m, ellipse3_3m, ellipse4_3m, ellipse5_3m, ellipse6_3m]

        # 6-meter buffer
        ellipse1_6m = Ellipse((760, 495), 400, 180, angle=0)
        ellipse2_6m = Ellipse((900, 355), 280, 80, angle=0)
        ellipse3_6m = Ellipse((1215, 630), 500, 280, angle=4)
        ellipse4_6m = Ellipse((730, 300), 220, 80, angle=0)
        ellipse5_6m = Ellipse((905, 320), 260, 80, angle=0)
        ellipse6_6m = Ellipse((1570, 560), 500, 200, angle=6)
        ellipses_data["6m"] = [ellipse1_6m, ellipse2_6m, ellipse3_6m, ellipse4_6m, ellipse5_6m, ellipse6_6m]

        # 9-meter buffer
        ellipse1_9m = Ellipse((760, 505), 660, 280, angle=0)
        ellipse2_9m = Ellipse((900, 355), 420, 115, angle=0)
        ellipse3_9m = Ellipse((1215, 665), 880, 400, angle=4)
        ellipse4_9m = Ellipse((730, 300), 330, 105, angle=0)
        ellipse5_9m = Ellipse((905, 320), 380, 100, angle=0)
        ellipse6_9m = Ellipse((1570, 590), 750, 380, angle=6)  
        ellipses_data["9m"] = [ellipse1_9m, ellipse2_9m, ellipse3_9m, ellipse4_9m, ellipse5_9m, ellipse6_9m]

    elif station == "TechwayB": 
        ellipses_data["1m"] = [Ellipse((1280, 490), 80, 20, angle=2), Ellipse((1020, 460), 40, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1280, 490), 260, 30, angle=2), Ellipse((1020, 460), 175, 25, angle=0)]
        ellipses_data["6m"] = [Ellipse((1280, 490), 500, 70, angle=2), Ellipse((1020, 465), 300, 30, angle=0)]
        ellipses_data["9m"] = [Ellipse((1280, 510), 740, 130, angle=2), Ellipse((1020, 465), 500, 45, angle=2)] 

    elif station == "TechwayC":
        # 1-meter buffer
        ellipses_data["1m"] = [Ellipse((1295, 470), 160, 35, angle=3.8), Ellipse((1000, 405), 60, 25, angle=2)]
        ellipses_data["3m"] = [Ellipse((1295, 480), 450, 80, angle=3.5), Ellipse((995, 415), 220, 30, angle=1.5)]
        ellipses_data["6m"] = [Ellipse((1295, 500), 700, 110, angle=3), Ellipse((995, 410), 350, 40, angle=1)]
        ellipses_data["9m"] = [Ellipse((1295, 510), 900, 230, angle=3), Ellipse((1000, 405), 500, 55, angle=1)]
        
    elif station == "TechwayD":   
        ellipses_data["1m"] = [Ellipse((335, 680), 65, 30, angle=0), Ellipse((570, 530), 35, 20, angle=1)]
        ellipses_data["3m"] = [Ellipse((335, 680), 210, 53, angle=1.0), Ellipse((570, 530), 150, 40, angle=1)]
        ellipses_data["6m"] = [Ellipse((345, 680), 480, 125, angle=1.0), Ellipse((585, 530), 280, 70, angle=1)]
        ellipses_data["9m"] = [Ellipse((375, 680), 620, 180, angle=1), Ellipse((595, 530), 400, 105, angle=1)]

    return(ellipses_data)

def ellipse_va_6072023(station):
    
    ellipses_data = {}
    
    if station == "Cadell": 
        # 1-meter buffer
        ellipse1_1m = Ellipse((790, 425), 70, 40, angle=0) 
        ellipse2_1m = Ellipse((930, 310), 70, 20, angle=0) 
        ellipse3_1m = Ellipse((1240, 550), 80, 40, angle=4) 
        ellipse4_1m = Ellipse((760, 260), 65, 20, angle=0) 
        ellipse5_1m = Ellipse((935, 275), 60, 20, angle=0) 
        ellipse6_1m = Ellipse((1610, 515), 90, 35, angle=4) 
        ellipses_data["1m"] = [ellipse1_1m, ellipse2_1m, ellipse3_1m, ellipse4_1m, ellipse5_1m, ellipse6_1m]

        # 3-meter buffer
        ellipse1_3m = Ellipse((790, 425), 200, 90, angle=0) # 790, 425
        ellipse2_3m = Ellipse((930, 310), 140, 40, angle=0) # 930, 310
        ellipse3_3m = Ellipse((1240, 550), 270, 140, angle=4) # 1240, 550
        ellipse4_3m = Ellipse((760, 260), 110, 40, angle=0) # 760, 260
        ellipse5_3m = Ellipse((935, 275), 120, 40, angle=0) # 935, 275
        ellipse6_3m = Ellipse((1610, 515), 250, 90, angle=6) # 1610, 515
        ellipses_data["3m"] = [ellipse1_3m, ellipse2_3m, ellipse3_3m, ellipse4_3m, ellipse5_3m, ellipse6_3m]

        # 6-meter buffer
        ellipse1_6m = Ellipse((790, 445), 400, 180, angle=0) # 790, 445
        ellipse2_6m = Ellipse((930, 315), 280, 80, angle=0) # 930, 315
        ellipse3_6m = Ellipse((1240, 600), 500, 280, angle=4) # 1240, 600
        ellipse4_6m = Ellipse((760, 260), 220, 80, angle=0) # 760, 260
        ellipse5_6m = Ellipse((935, 280), 260, 80, angle=0) # 935, 280
        ellipse6_6m = Ellipse((1610, 535), 500, 200, angle=6) # 1610, 535
        ellipses_data["6m"] = [ellipse1_6m, ellipse2_6m, ellipse3_6m, ellipse4_6m, ellipse5_6m, ellipse6_6m]

        # 9-meter buffer
        ellipse1_9m = Ellipse((790, 455), 660, 280, angle=0) # 790, 455
        ellipse2_9m = Ellipse((930, 310), 420, 115, angle=0) # 930, 310
        ellipse3_9m = Ellipse((1240, 635), 880, 400, angle=4) # 1240, 635
        ellipse4_9m = Ellipse((760, 260), 330, 105, angle=0) # 760, 260
        ellipse5_9m = Ellipse((935, 280), 380, 100, angle=0) # 935, 280
        ellipse6_9m = Ellipse((1610, 565), 750, 380, angle=6) # 1610, 565  
        ellipses_data["9m"] = [ellipse1_9m, ellipse2_9m, ellipse3_9m, ellipse4_9m, ellipse5_9m, ellipse6_9m]
        
    if station == "TechwayA": 
        ellipses_data["1m"] = [Ellipse((745, 425), 180, 50, angle=2), Ellipse((405, 390), 65, 20, angle=0), Ellipse((300, 390), 50, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((745, 435), 380, 100, angle=-2), Ellipse((405, 390), 240, 40, angle=-2), Ellipse((300, 390), 170, 20, angle=-2)]
        ellipses_data["6m"] = [Ellipse((765, 465), 760, 210, angle=-1), Ellipse((400, 400), 460, 60, angle=-2), Ellipse((295, 390), 360, 28, angle=-2)]
        ellipses_data["9m"] = [Ellipse((765, 505), 1180, 330, angle=-2), Ellipse((410, 410), 650, 80, angle=-2), Ellipse((300, 390), 450, 30, angle=-2)]

    elif station == "TechwayB":
        ellipses_data["1m"] = [Ellipse((1355, 465), 80, 20, angle=2), Ellipse((1090, 400), 40, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1355, 460), 260, 30, angle=2), Ellipse((1090, 400), 175, 25, angle=0)]
        ellipses_data["6m"] = [Ellipse((1345, 460), 520, 70, angle=2), Ellipse((1090, 405), 300, 30, angle=0)]
        ellipses_data["9m"] = [Ellipse((1335, 480), 850, 130, angle=2), Ellipse((1090, 405), 500, 45, angle=2)]

    elif station == "TechwayD":
        ellipses_data["1m"] = [Ellipse((365, 490), 65, 30, angle=0), Ellipse((600, 330), 35, 20, angle=1)]
        ellipses_data["3m"] = [Ellipse((365, 490), 210, 53, angle=1.0), Ellipse((600, 330), 150, 40, angle=1)]
        ellipses_data["6m"] = [Ellipse((375, 490), 480, 125, angle=1.0), Ellipse((615, 330), 280, 70, angle=1)]
        ellipses_data["9m"] = [Ellipse((410, 490), 620, 180, angle=1), Ellipse((625, 330), 400, 105, angle=1)]
        
    return(ellipses_data)

def ellipse_va_6212023(station):
    
    ellipses_data = {}
    
    if station == "Cadell": 
        # 1-meter buffer
        ellipse1_1m = Ellipse((775, 305), 70, 40, angle=0) # 775, 305
        ellipse2_1m = Ellipse((910, 185), 70, 20, angle=0) # 910, 185
        ellipse3_1m = Ellipse((1235, 415), 80, 40, angle=4) # 1235, 415
        ellipse4_1m = Ellipse((705, 165), 65, 20, angle=0) # 705, 165
        ellipse5_1m = Ellipse((920, 155), 60, 20, angle=0) # 920, 155
        ellipse6_1m = Ellipse((1600, 390), 90, 35, angle=4) # 1600, 390
        ellipses_data["1m"] = [ellipse1_1m, ellipse2_1m, ellipse3_1m, ellipse4_1m, ellipse5_1m, ellipse6_1m]


        # 3-meter buffer
        ellipse1_3m = Ellipse((775, 305), 200, 90, angle=0) # 775, 305
        ellipse2_3m = Ellipse((910, 185), 140, 40, angle=0) # 910, 185
        ellipse3_3m = Ellipse((1235, 415), 270, 140, angle=4) # 1235, 415
        ellipse4_3m = Ellipse((705, 165), 110, 40, angle=0) # 705, 165
        ellipse5_3m = Ellipse((920, 155), 120, 40, angle=0) # 920, 155
        ellipse6_3m = Ellipse((1600, 390), 250, 90, angle=6) # 1600, 390
        ellipses_data["3m"] = [ellipse1_3m, ellipse2_3m, ellipse3_3m, ellipse4_3m, ellipse5_3m, ellipse6_3m]

        # 6-meter buffer
        ellipse1_6m = Ellipse((775, 325), 400, 180, angle=0) # 775, 325
        ellipse2_6m = Ellipse((910, 190), 280, 80, angle=0) # 910, 190
        ellipse3_6m = Ellipse((1235, 465), 500, 280, angle=4) # 1235, 465
        ellipse4_6m = Ellipse((705, 165), 220, 80, angle=0) # 705, 165
        ellipse5_6m = Ellipse((920, 160), 260, 80, angle=0) # 920, 160
        ellipse6_6m = Ellipse((1600, 410), 500, 200, angle=6) # 1600, 410
        ellipses_data["6m"] = [ellipse1_6m, ellipse2_6m, ellipse3_6m, ellipse4_6m, ellipse5_6m, ellipse6_6m]

        # 9-meter buffer
        ellipse1_9m = Ellipse((775, 335), 660, 280, angle=0) # 775, 335
        ellipse2_9m = Ellipse((910, 190), 420, 115, angle=0) # 910, 190
        ellipse3_9m = Ellipse((1235, 500), 880, 400, angle=4) # 1235, 500
        ellipse4_9m = Ellipse((705, 165), 330, 105, angle=0) # 705, 165
        ellipse5_9m = Ellipse((920, 160), 380, 100, angle=0) # 920, 160
        ellipse6_9m = Ellipse((1600, 440), 750, 380, angle=6) # 1600, 440
        ellipses_data["9m"] = [ellipse1_9m, ellipse2_9m, ellipse3_9m, ellipse4_9m, ellipse5_9m, ellipse6_9m]
        
    elif station == "TechwayA":
        ellipses_data["1m"] = [Ellipse((1060, 430), 180, 50, angle=2), Ellipse((735, 370), 65, 20, angle=0), Ellipse((635, 360), 50, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1060, 435), 380, 100, angle=-2), Ellipse((735, 370), 240, 40, angle=-2), Ellipse((635, 360), 170, 20, angle=-2)]
        ellipses_data["6m"] = [Ellipse((1080, 460), 760, 210, angle=-1), Ellipse((730, 370), 460, 60, angle=-2), Ellipse((630, 360), 360, 28, angle=-2)]
        ellipses_data["9m"] = [Ellipse((1080, 500), 1180, 330, angle=-2), Ellipse((735, 380), 650, 80, angle=-2), Ellipse((635, 360), 450, 30, angle=-2)]

    elif station == "TechwayB":
        ellipses_data["1m"] = [Ellipse((1270, 500), 80, 20, angle=2), Ellipse((1015, 445), 40, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1270, 500), 260, 30, angle=2), Ellipse((1015, 445), 175, 25, angle=0)]
        ellipses_data["6m"] = [Ellipse((1270, 500), 550, 70, angle=2), Ellipse((1015, 450), 330, 30, angle=0)]
        ellipses_data["9m"] = [Ellipse((1270, 520), 900, 130, angle=2), Ellipse((1015, 450), 500, 45, angle=2)]

    elif station == "TechwayC":
        ellipses_data["1m"] = [Ellipse((1645, 645), 160, 35, angle=3.8), Ellipse((1345, 560), 60, 25, angle=2)]
        ellipses_data["3m"] = [Ellipse((1645, 655), 450, 80, angle=3.5), Ellipse((1340, 570), 220, 30, angle=1.5)]
        ellipses_data["6m"] = [Ellipse((1645, 675), 900, 110, angle=3), Ellipse((1340, 565), 350, 40, angle=1)]
        ellipses_data["9m"] = [Ellipse((1645, 685), 1200, 230, angle=3), Ellipse((1345, 560), 600, 55, angle=1)]

    elif station == "TechwayD":
        ellipses_data["1m"] = [Ellipse((350, 660), 65, 30, angle=0), Ellipse((585, 510), 35, 20, angle=1)]
        ellipses_data["3m"] = [Ellipse((350, 660), 210, 53, angle=1.0), Ellipse((585, 510), 150, 40, angle=1)]
        ellipses_data["6m"] = [Ellipse((360, 660), 480, 125, angle=1.0), Ellipse((600, 510), 280, 70, angle=1)]
        ellipses_data["9m"] = [Ellipse((395, 660), 620, 180, angle=1), Ellipse((605, 510), 400, 105, angle=1)]

    return(ellipses_data)

def ellipse_va_6282023(station):
    
    ellipses_data = {}
    
    if station == "Cadell": 
        # 1-meter buffer
        ellipse1_1m = Ellipse((770, 335), 70, 40, angle=0) # 770, 335
        ellipse2_1m = Ellipse((905, 220), 70, 20, angle=0) # 910, 220
        ellipse3_1m = Ellipse((1230, 455), 80, 40, angle=4) # 1230, 455
        ellipse4_1m = Ellipse((710, 190), 65, 20, angle=0) # 710, 190
        ellipse5_1m = Ellipse((915, 185), 60, 20, angle=0) # 915, 185
        ellipse6_1m = Ellipse((1590, 430), 90, 35, angle=4) # 1590, 430
        ellipses_data["1m"] = [ellipse1_1m, ellipse2_1m, ellipse3_1m, ellipse4_1m, ellipse5_1m, ellipse6_1m]


        # 3-meter buffer
        ellipse1_3m = Ellipse((770, 335), 200, 90, angle=0) # 770, 335
        ellipse2_3m = Ellipse((905, 220), 140, 40, angle=0) # 910, 220
        ellipse3_3m = Ellipse((1230, 455), 270, 140, angle=4) # 1230, 455
        ellipse4_3m = Ellipse((710, 190), 110, 40, angle=0) # 710, 190
        ellipse5_3m = Ellipse((915, 185), 120, 40, angle=0) # 915, 185
        ellipse6_3m = Ellipse((1590, 430), 250, 90, angle=6) # 1590, 430
        ellipses_data["3m"] = [ellipse1_3m, ellipse2_3m, ellipse3_3m, ellipse4_3m, ellipse5_3m, ellipse6_3m]

        # 6-meter buffer
        ellipse1_6m = Ellipse((770, 355), 400, 180, angle=0) # 770, 355
        ellipse2_6m = Ellipse((905, 225), 280, 80, angle=0) # 910, 225
        ellipse3_6m = Ellipse((1230, 505), 500, 280, angle=4) # 1230, 505
        ellipse4_6m = Ellipse((710, 190), 220, 80, angle=0) # 710, 190
        ellipse5_6m = Ellipse((915, 195), 260, 80, angle=0) # 915, 195
        ellipse6_6m = Ellipse((1590, 450), 500, 200, angle=6) # 1590, 450
        ellipses_data["6m"] = [ellipse1_6m, ellipse2_6m, ellipse3_6m, ellipse4_6m, ellipse5_6m, ellipse6_6m]

        # 9-meter buffer
        ellipse1_9m = Ellipse((770, 365), 660, 280, angle=0) # 770, 365
        ellipse2_9m = Ellipse((905, 225), 420, 115, angle=0) # 910, 225
        ellipse3_9m = Ellipse((1230, 535), 880, 400, angle=4) # 1230, 535
        ellipse4_9m = Ellipse((710, 190), 330, 105, angle=0) # 710, 190
        ellipse5_9m = Ellipse((915, 195), 380, 100, angle=0) # 915, 195
        ellipse6_9m = Ellipse((1590, 480), 750, 380, angle=6) # 1590, 480
        ellipses_data["9m"] = [ellipse1_9m, ellipse2_9m, ellipse3_9m, ellipse4_9m, ellipse5_9m, ellipse6_9m]
        
    elif station == "TechwayA":
        ellipses_data["1m"] = [Ellipse((805, 480), 180, 50, angle=2), Ellipse((470, 435), 65, 20, angle=0), Ellipse((370, 430), 50, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((805, 490), 380, 100, angle=-2), Ellipse((470, 435), 240, 40, angle=-2), Ellipse((370, 430), 170, 20, angle=-2)]
        ellipses_data["6m"] = [Ellipse((825, 520), 760, 210, angle=-1), Ellipse((465, 445), 460, 60, angle=-2), Ellipse((365, 430), 360, 28, angle=-2)]
        ellipses_data["9m"] = [Ellipse((825, 560), 1180, 330, angle=-2), Ellipse((470, 455), 650, 80, angle=-2), Ellipse((370, 430), 450, 30, angle=-2)]
        
    elif station == "TechwayB":
        ellipses_data["1m"] = [Ellipse((1315, 470), 80, 20, angle=2), Ellipse((1055, 425), 40, 15, angle=0)]
        ellipses_data["3m"] = [Ellipse((1315, 470), 260, 30, angle=2), Ellipse((1055, 425), 175, 25, angle=0)]
        ellipses_data["6m"] = [Ellipse((1315, 470), 520, 70, angle=2), Ellipse((1055, 430), 350, 30, angle=0)]
        ellipses_data["9m"] = [Ellipse((1315, 490), 760, 130, angle=2), Ellipse((1055, 430), 500, 45, angle=2)]

    elif station == "TechwayD":
        ellipses_data["1m"] = [Ellipse((680, 850), 65, 30, angle=0), Ellipse((930, 730), 35, 20, angle=1)]
        ellipses_data["3m"] = [Ellipse((680, 850), 210, 53, angle=1.0), Ellipse((930, 730), 150, 40, angle=1)]
        ellipses_data["6m"] = [Ellipse((690, 850), 480, 125, angle=1.0), Ellipse((945, 730), 280, 70, angle=1)]
        ellipses_data["9m"] = [Ellipse((725, 850), 620, 180, angle=1), Ellipse((955, 730), 400, 105, angle=1)]

    return(ellipses_data)