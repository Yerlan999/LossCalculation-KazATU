class MainTrackerClass():

    xy_grid_list = []
    mat_prop_grid_list = []
    rad_but_list = []
    all_entries = []
    xy_labels = []

    image_set = False
    current_image = None

    all_images_paths = {
        1: "pics/VL1.jpg",
        11: "pics/VL1_mir.jpg",
        2: "pics/VL2.jpg",
        3: "pics/VL3.jpg",
    }
    xy_headers = ["X", "Y"]

    xyframe_label_dict = {
        3 : ["T1","T2","A","B","C"],
        1 : ["T","A","B","C"],
        2 : ["T","A1","B1","C1","A2","B2","C2"],
    }
    xy_var_list = []



    mat_headers = ["Фаза", "Трос"]
    mat_props = ["Маг.проницаемость (\u03bc)", "Уд.проводимость (\u03C3)", "Попер.сечение (S)"]

    types_cables = ["inp_faza_", "inp_tros_"]
    mater_charac = ["mag", "gam", "pop"]
    mat_var_list = []


    main_entry_list_names = [

        ("inp_name_podstans", "Наименование подстанции", "", ()),
        ("inp_name_prisoedin", "Наименования присоед-ии", "", ()),
        ("inp_kolich_prisoed", "Количество присоед.", "", ()),
        ("inp_dlina_linii", "Длина линии", 'dlina_edin_izmer', ("м", "км")),
        ("inp_kolich_izmer", "Количество измерении", "", ()),
        ("inp_interval_izmer", "Интервал измерении", 'inter_izmer_edin_izmer', ("сек", "мин")),
        ("inp_kolich_garmonik", "Количество гармоник", "", ()),
        ("inp_napryazhen_linii", "Напряжение линии", 'naprzh_linii_edin_izmer', ("В", "кВ"))]

    main_entry_list_vars = []

    edin_imzer_list_names =[

        ('dlina_edin_izmer', 'км'),
        ('inter_izmer_edin_izmer', 'мин'),
        ('naprzh_linii_edin_izmer', 'кВ'),
        ('gamma_edin_izmer', 'МСм/м'),
        ('pop_sech_edin_izmer', 'мм\u00b2')];

    edin_imzer_list_vars = []

    mat_edin_izmer = [
        ("gamma_edin_izmer", ("См/м", "кСм/м", "МСм/м")),
        ("pop_sech_edin_izmer", ("мм\u00b2", "см\u00b2", "м\u00b2")),
    ]

    list_of_frames = []
    current_frame = None

    excel_file_path = None

    mirror_button = None
    mirror_state = 1

