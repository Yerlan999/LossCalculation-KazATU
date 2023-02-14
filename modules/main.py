import shutil, codecs, re, os, threading
from time import sleep
import os
import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox, HORIZONTAL, END
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image
from pathlib import Path
from tkinter import ttk
from ttkthemes import ThemedTk


class MainTrackerClass():

    xy_grid_list = []
    mat_prop_grid_list = []
    rad_but_list = []
    all_entries = []
    xy_labels = []
    label_started = dict()
    destroyables = []

    image_set = False
    current_image = None

    all_images_paths = {
        1: "pics/VL1.jpg",
        11: "pics/VL1_mir.jpg",
        2: "pics/VL2.jpg",
    }
    xy_headers = ["X", "Y"]

    xyframe_label_dict = {
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
        ("inp_dlina_linii", "Длина линии", 'dlina_edin_izmer', ("м", "км")),
        ("inp_kolich_izmer", "Количество измерении", "", ()),
        ("inp_interval_izmer", "Интервал измерении", 'inter_izmer_edin_izmer', ("сек", "мин")),
        ("inp_kolich_garmonik", "Количество гармоник", "", ()),
    ]

    main_entry_list_vars = []

    edin_imzer_list_names =[

        ('dlina_edin_izmer', 'км'),
        ('inter_izmer_edin_izmer', 'мин'),
        ('gamma_edin_izmer', 'МСм/м'),
        ('pop_sech_edin_izmer', 'мм\u00b2')];

    edin_imzer_list_vars = []

    mat_edin_izmer = [
        ("gamma_edin_izmer", ("См/м", "кСм/м", "МСм/м")),
        ("pop_sech_edin_izmer", ("мм\u00b2", "см\u00b2", "м\u00b2")),
    ]

    list_of_frames = []
    current_frame = None

    # excel_file_path = StringVar()

    mirror_button = None
    mirror_state = 1



def mirror_picture(canvas):

    if MainTrackerClass.mirror_state == 1:
        img = Image.open(resource_path(Path(MainTrackerClass.all_images_paths[11])))
        resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                                 Image.BICUBIC)
        img = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, anchor=NW, image=img)
        MainTrackerClass.current_image = 11
        MainTrackerClass.mirror_state = 2
    else:
        img = Image.open(resource_path(Path(MainTrackerClass.all_images_paths[1])))
        resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                                 Image.BICUBIC)
        img = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, anchor=NW, image=img)
        MainTrackerClass.current_image = 1
        MainTrackerClass.mirror_state = 1

    xy_frame.mainloop()


def on_closing():
    if messagebox.askokcancel("Выход из программы", "Вы действительно хотите выйти?"):
        root.destroy()


def help_func():

    help_root = ThemedTk(theme=theme)
    help_root.geometry("600x200")
    help_frame = ttk.Frame(help_root)

    ttk.Label(help_frame, text="Для выполнения расчетов необходимо заполнить все поля приведенные в двух разделах!", width=100, anchor=CENTER, borderwidth=2, relief="groove").pack(side=TOP, fill=BOTH, expand=True)
    ttk.Label(help_frame, text="Иконку программы необходимо располагать рядом с папкой 'pics'", width=100, anchor=CENTER, borderwidth=2, relief="groove").pack(side=BOTTOM, fill=BOTH, expand=True)
    ttk.Label(help_frame, text="Все разделы находятся в меню 'Входные данные'", width=100, anchor=CENTER, borderwidth=2, relief="groove").pack(side=BOTTOM, fill=BOTH, expand=True)
    help_frame.pack(side=LEFT, fill=BOTH, expand=True)
    help_root.mainloop()


def validate_numbers(index, numbers):
    return globals()["pattern"].match(numbers) is not None

def on_click(event):
    global eb, on_click_id
    eb.delete(0, END)
    eb.unbind('<Button-1>', on_click_id)



def finishing_part(name_podstans):
    inp = name_podstans.strip().replace(" ", "_")
    dir_to_check = f"Результаты_{inp}"
    pa = Path(dir_to_check)
    try:
        if pa.is_dir():
            os.chdir(dir_to_check)
            with codecs.open(f"{dir_to_check.lower()}.txt", "w", "utf-16") as file:
                text_message = u"Здесь будут результаты расчетов(графики) после того как я разберусь с 'Телеграфными уравнениями')))"
                file.write(text_message + u"\n")
        else:
            os.mkdir(dir_to_check)
            os.chdir(dir_to_check)
            with codecs.open(f"{dir_to_check.lower()}.txt", "w", "utf-16") as file:
                text_message = u"Здесь будут результаты расчетов(графики) после того как я разберусь с 'Телеграфными уравнениями')))"
                file.write(text_message + u"\n")
    finally:
        os.chdir(Path("../"))



def subbmit_values(MainTrackerClass):

    answer = messagebox.askyesno(title="Расчет", message="Вы уверены что хотите произвести раcчет по указанным данным?")
    if answer == False: return;

    def submit_button():

        # Extracting and preprocessing user input
        try:
            excel_filepath = excel_file_path.get()
            excel_filepath_os = Path(excel_filepath)

            name_podstans = MainTrackerClass.main_entry_list_vars[0].get().strip()
            dlina_linii = float(MainTrackerClass.main_entry_list_vars[1].get())
            kolich_izmer = float(MainTrackerClass.main_entry_list_vars[2].get())
            interval_izmer = float(MainTrackerClass.main_entry_list_vars[3].get())
            kolich_garmonik = float(MainTrackerClass.main_entry_list_vars[4].get())


            if MainTrackerClass.edin_imzer_list_vars[0].get() == "м":
                dlina_linii = dlina_linii/1000
            if MainTrackerClass.edin_imzer_list_vars[1].get() == 'сек':
                interval_izmer = interval_izmer/60

        except Exception as error:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных воздушной линии")
            main_entry_list_vars = None
            return

        if MainTrackerClass.current_image == 1:
            tx, ty, ax, ay, bx, by, cx, cy =  MainTrackerClass.xy_grid_list
            list_of_xys = [tx, ty, ax, ay, bx, by, cx, cy]
        if MainTrackerClass.current_image == 2:
            tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y =  MainTrackerClass.xy_grid_list
            list_of_xys = [tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y]

        getted_list_xys = list(map(lambda e: e.get(), list_of_xys))

        try:
            floated_list_xys = list(map(lambda e: float(e), getted_list_xys))
        except:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для координат фазных проводов и троса")
            return

        # Module for Material Prop. data
        getted_list_matprop = list(map(lambda e: e.get(), MainTrackerClass.mat_var_list))

        # Unit conversions happens here
        try:
            floated_list_matprop = list(map(lambda e: float(e), getted_list_matprop))
            if MainTrackerClass.edin_imzer_list_vars[2].get() == "кСм/м":
                floated_list_matprop[1] = floated_list_matprop[1]*10**3
                floated_list_matprop[4] = floated_list_matprop[4]*10**3
            if MainTrackerClass.edin_imzer_list_vars[2].get() == "См/м":
                floated_list_matprop[1] = floated_list_matprop[1]*10**6
                floated_list_matprop[4] = floated_list_matprop[4]*10**6

            if MainTrackerClass.edin_imzer_list_vars[3].get() == "м\u00b2":
                floated_list_matprop[2] = floated_list_matprop[2]*10**6
                floated_list_matprop[5] = floated_list_matprop[5]*10**6
            if MainTrackerClass.edin_imzer_list_vars[3].get() == "см\u00b2":
                floated_list_matprop[2] = floated_list_matprop[2]*100
                floated_list_matprop[5] = floated_list_matprop[5]*100
        except:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для характеристик материала фазных проводов и троса")
            return


        print()
        print(MainTrackerClass.label_started[prisoed_to_calc.get()])
        prisoed_to_calc.set('')

        MainTrackerClass.current_frame.pack_forget()
        MainTrackerClass.current_frame = None

        progress_frame = ttk.Frame(root)

        progress_frame.rowconfigure(list(range(0,50)), weight=1)
        progress_frame.columnconfigure(list(range(0,11)), weight=1)

        progress_bar = Progressbar(progress_frame, orient=HORIZONTAL, mode='indeterminate')
        progress_bar.grid(row=20, column=3, columnspan=5, sticky="EWNS")
        progress_bar_label = ttk.Label(progress_frame, text="Обработка данных...", anchor=CENTER, borderwidth=2, relief="groove")
        progress_bar_label.grid(row=19, column=3, columnspan=5, sticky="EWNS")
        progress_bar.start()
        progress_frame.pack(side=BOTTOM, anchor=E, fill=BOTH, expand=True)

        # Clear all entries
        for entry in MainTrackerClass.all_entries + MainTrackerClass.xy_grid_list:
            try:
                entry.delete(0, 'end')
            except Exception as error:
                print("Something went wrong!", error)
                pass
        finishing_part(name_podstans)
        final_message = f"Результаты расчетов записаны в папке 'Результаты_{name_podstans}'"


        sleep(5) # EMULATING CALCULATION PROCESS
        progress_bar.stop()
        progress_bar_label.destroy()
        progress_bar.destroy()
        progress_frame.destroy()
        messagebox.showinfo(title="Расчет завершен!", message="Данные успешно обработаны!", detail=final_message)
        #root.destroy()
        main_properties(MainTrackerClass)

    threading.Thread(target=submit_button).start()




def main_properties(main):

    global eb, on_click_id

    if MainTrackerClass.current_frame:
        MainTrackerClass.current_frame.pack_forget()
    MainTrackerClass.current_frame = main_frame

    ttk.Label(main_frame, text="Общие данные линии", width=55, anchor=CENTER, borderwidth=2, relief="groove", font=('Helvetica', 13)).grid(row=0, column=0, columnspan=4, sticky="EWNS")

    for i, (variable, label, var, ed_iz) in enumerate(MainTrackerClass.main_entry_list_names, start=1):

        if i == 1:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            MainTrackerClass.all_entries.append(e)
        elif ed_iz:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, validate="key", validatecommand=vcmd, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            MainTrackerClass.all_entries.append(e)
            OptionMenu(main_frame, eval(var), *ed_iz).grid(row=i, column=3, sticky="EWNS")
        else:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, validate="key", validatecommand=vcmd, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            MainTrackerClass.all_entries.append(e)


    ttk.Label(main_frame, text="Характеристика проводов", width=55, anchor=CENTER, borderwidth=2, relief="groove", font=('Helvetica', 13)).grid(row=10, column=0, columnspan=4, sticky="EWNS")

    for i, mat_lab in enumerate(MainTrackerClass.mat_props, start=12):
        ttk.Label(main_frame, text=mat_lab, width=15, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")


    for i, mat_head in enumerate(MainTrackerClass.mat_headers, start=1):
        ttk.Label(main_frame, text=mat_head, width=15, anchor=CENTER, borderwidth=2, relief="groove").grid(row=11, column=i, sticky="EWNS")

    for j, provod in enumerate(MainTrackerClass.types_cables, start=1):
        for i, character in enumerate(MainTrackerClass.mater_charac, start=12):
            me=ttk.Entry(main_frame, width=20, textvariable=eval(provod+character), validate="key", validatecommand=vcmd); me.grid(row=i, column=j, sticky="EWNS")
            MainTrackerClass.all_entries.append(me)

    OptionMenu(main_frame, eval(MainTrackerClass.mat_edin_izmer[0][0]), *MainTrackerClass.mat_edin_izmer[0][1]).grid(row=13, column=3, sticky="EWNS")
    OptionMenu(main_frame, eval(MainTrackerClass.mat_edin_izmer[1][0]), *MainTrackerClass.mat_edin_izmer[1][1]).grid(row=14, column=3, sticky="EWNS")


    main_frame.pack(side=TOP, anchor=E, fill=BOTH, expand=True)


def resource_path(relative_path):
    try:
        base_path = Path(".")
    except Exception:
        base_path = sys._MEIPASS
    return base_path / relative_path




def draw_picture(canvas):
    global image_set, mirror_button, previous_image
    canvas.delete("all")

    MainTrackerClass.image_set = True
    MainTrackerClass.current_image = frame.get()

    if MainTrackerClass.current_image == 1:
        MainTrackerClass.mirror_button["state"] = "active"
    else:
        MainTrackerClass.mirror_button["state"] = "disabled"

    img = Image.open(resource_path(Path(MainTrackerClass.all_images_paths[frame.get()])))
    resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                             Image.BICUBIC)
    img = ImageTk.PhotoImage(resized)
    canvas.create_image(0, 0, anchor=NW, image=img)


    ttk.Label(xy_frame, text="Y", width=1, anchor=CENTER, borderwidth=2, relief="groove").grid(row=1, column=3, sticky="EWNS")
    ttk.Label(xy_frame, text="X", width=1, anchor=CENTER, borderwidth=2, relief="groove").grid(row=1, column=2, sticky="EWNS")


    # ROPE LABELS
    for cur_lab in MainTrackerClass.xy_labels:
        if cur_lab:
            cur_lab.destroy()
    MainTrackerClass.xy_labels= []


    for i, label in enumerate(MainTrackerClass.xyframe_label_dict[frame.get()], start=2):
        l=ttk.Label(xy_frame, text=label, width=4, anchor=CENTER, borderwidth=2, relief="groove"); l.grid(row=i, column=1, sticky="EWNS")
        MainTrackerClass.xy_labels.append(l)


    # XY ENTRIES
    if previous_image != MainTrackerClass.current_image:

        for xy_curr_entries in MainTrackerClass.xy_grid_list:
            if xy_curr_entries:
                xy_curr_entries.destroy()
        MainTrackerClass.xy_grid_list = []

        for i, rope in enumerate(MainTrackerClass.xyframe_label_dict[frame.get()], start=2):
            for j, xy in enumerate(MainTrackerClass.xy_headers, start=2):
                inp_var = (rope+xy).lower()
                exec(inp_var + "=StringVar()")
                MainTrackerClass.xy_var_list.append(eval(inp_var))
                xye=ttk.Entry(xy_frame, textvariable=eval(inp_var), width=1, validate="key", validatecommand=vcmd); xye.grid(row=i, column=j, sticky="EWNS")
                MainTrackerClass.xy_grid_list.append(xye)


    previous_image = MainTrackerClass.current_image

    xy_frame.mainloop()



def xy_properties(main):
    global canvas

    if MainTrackerClass.current_frame:
        MainTrackerClass.current_frame.pack_forget()
    MainTrackerClass.current_frame = xy_frame

    ttk.Label(xy_frame, text="Характеристика опор", width=75, anchor=CENTER, borderwidth=2, relief="groove", font=('Helvetica', 13)).grid(row=0, column=0, columnspan=4, sticky="EWNS")

    if not frame.get():
        canvas = Canvas(xy_frame, width = 281, height = 313, bg='white')
        canvas.grid(row=1, rowspan=9, column=0, sticky="EWNS")

    MainTrackerClass.mirror_button = ttk.Button(xy_frame, text="<>", command=lambda:mirror_picture(canvas), cursor='hand2')
    MainTrackerClass.mirror_button.grid(row=9, column=0, sticky="SE")
    MainTrackerClass.mirror_button["state"] = "disabled"

    ttk.Label(xy_frame, text="Тип опоры", width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=10, column=0, columnspan=1, sticky="EWNS")


    def resize_bg(event):
        global bgg, resized, bg2, canvas
        if not MainTrackerClass.image_set:
            return
        canvas.delete("all")

        bgg = Image.open(MainTrackerClass.all_images_paths[MainTrackerClass.current_image])
        resized = bgg.resize((canvas.winfo_width(), canvas.winfo_height()),
                             Image.BICUBIC)

        bg2 = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, image=bg2, anchor='nw')

    ttk.Radiobutton(xy_frame, text="Одноцепная промежуточная", variable=frame, value=1, command=lambda:draw_picture(canvas), cursor='hand2').grid(row=11, column=0, sticky="WNS")
    ttk.Radiobutton(xy_frame, text="Двухцепная промежуточная", variable=frame, value=2, command=lambda:draw_picture(canvas), cursor='hand2').grid(row=12, column=0, sticky="WNS")

    def fileinput():
        filename = filedialog.askopenfilename(filetypes=[("Файлы EXCEL", ".xlsx .xls"), ("Все файлы","*.*")])
        excel_filepath.insert(END, filename)

        def read_excel(excel_filepath, xy_frame):
            progress_frame = ttk.Frame(root)

            progress_frame.rowconfigure(list(range(0,50)), weight=1)
            progress_frame.columnconfigure(list(range(0,11)), weight=1)

            progress_bar = Progressbar(progress_frame, orient=HORIZONTAL, mode='indeterminate')
            progress_bar.grid(row=20, column=3, columnspan=5, sticky="EWNS")
            progress_bar_label = ttk.Label(progress_frame, text="Чтение файла...", anchor=CENTER, borderwidth=2, relief="groove")
            progress_bar_label.grid(row=19, column=3, columnspan=5, sticky="EWNS")
            progress_bar.start()
            progress_frame.pack(side=BOTTOM, anchor=E, fill=BOTH, expand=True)

            print(excel_filepath.get())
            excel_file = pd.ExcelFile(excel_filepath.get())

            dataframe = pd.read_excel(excel_file, excel_file.sheet_names[0], header=None)
            label_started = dict()

            label_count = 0;
            for i, row in enumerate(dataframe.iloc[:,0]):
                if type(row) == str:
                    label_count += 1;
                    label_started[row] = [label_count, i]
                else:
                    continue

            for i, item in enumerate(label_started.keys()):
                label_started[item][1] -= i

            MainTrackerClass.label_started = label_started

            label_excel = ttk.Label(xy_frame, text="Выбрать присоединение для расчета", width=25, anchor=CENTER, borderwidth=2, relief="groove")
            label_excel.grid(row=15, column=0, sticky="EWNS", columnspan=4)

            prises = list(label_started.keys())
            prises.append("Все")

            OptionMenu(xy_frame, prisoed_to_calc, *prises).grid(row=16, column=0, columnspan=4, sticky="EWNS")

            progress_bar.stop()
            progress_bar_label.destroy()
            progress_bar.destroy()
            progress_frame.destroy()

        threading.Thread(target=read_excel, args=(excel_filepath, xy_frame)).start()


    label_excel = ttk.Label(xy_frame, text="Путь к EXCEL файлу", width=25, anchor=CENTER, borderwidth=2, relief="groove")
    label_excel.grid(row=14, column=0, sticky="EWNS")

    # MainTrackerClass.excel_file_path = StringVar()
    excel_filepath = ttk.Entry(xy_frame, width=5, textvariable=excel_file_path)
    MainTrackerClass.all_entries.append(excel_filepath)
    excel_filepath.grid(row=14, column=1, sticky="EWNS")

    select_button_excel = ttk.Button(xy_frame, text="Выбрать", command=fileinput, cursor='hand2')
    select_button_excel.grid(row=14, column=2, columnspan=2, sticky="EWNS")

    xy_frame.bind("<Configure>", resize_bg)
    xy_frame.pack(side=TOP, anchor=E, fill=BOTH, expand=True)
    xy_frame.mainloop()

MainTrackerClass.image_set = False

theme = 'breeze'

root = ThemedTk(theme=theme)
root.geometry("600x600+0+0")


pattern = re.compile(r'^([\.\d]*)$')
vcmd = (root.register(validate_numbers), "%i", "%P")



frame = IntVar(); frame.set(0)
previous_image = frame.get()




main_frame = ttk.Frame(root)
main_frame.rowconfigure(list(range(0,17)), weight=2)
main_frame.columnconfigure(list(range(0,4)), weight=1)
main_frame.columnconfigure(3, weight=0)


xy_frame = ttk.Frame(root)
xy_frame.rowconfigure(list(range(0,18)), weight=2)
xy_frame.columnconfigure(list(range(1,4)), weight=1)
xy_frame.columnconfigure(0, weight=2)


MainTrackerClass.list_of_frames = [xy_frame, main_frame]


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Общие характеристики линии", command=lambda:main_properties(MainTrackerClass))
filemenu.add_command(label="Характеристика опоры", command=lambda:xy_properties(MainTrackerClass))
filemenu.add_separator()
filemenu.add_command(label="Рассчитать потери", command=lambda:subbmit_values(MainTrackerClass))
menubar.add_cascade(label="Входные данные", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Инструкция программы", command=help_func)
helpmenu.add_command(label="Выйти из программы", command=on_closing)
menubar.add_cascade(label="О программе", menu=helpmenu)

root.config(menu=menubar)

root.protocol("WM_DELETE_WINDOW", on_closing)

for main_var, *rest in MainTrackerClass.main_entry_list_names:
    exec(main_var + "=StringVar()")
    MainTrackerClass.main_entry_list_vars.append(eval(main_var))


for main_var, def_val in MainTrackerClass.edin_imzer_list_names:
    exec(main_var + "=StringVar()")
    eval(main_var).set(def_val)
    MainTrackerClass.edin_imzer_list_vars.append(eval(main_var))


for provod in MainTrackerClass.types_cables:
    for character in MainTrackerClass.mater_charac:
        exec(provod+character + "=StringVar()")
        MainTrackerClass.mat_var_list.append(eval(provod+character))

excel_file_path = StringVar()
prisoed_to_calc = StringVar()

root.mainloop()

