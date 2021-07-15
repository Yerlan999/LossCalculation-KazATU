from time import sleep
import shutil, codecs, re, os, threading
from tkinter import *
from PIL import ImageTk,Image
from tkinter.ttk import Progressbar
from tkinter import filedialog, messagebox, HORIZONTAL, END
from pathlib import Path


label_color = "#525a53"
window_color = "#bdb7b1"
entry_color = "#e5e5e5"
text_color = "#ffffff"

main_label_color = "#bdb7b1"
main_text_color = "#000000"

calc_button_color_1 = "#274c77"
calc_button_color_2 = "#14213d"


class SubmitProgressBar():
    def __init__(self):
        self.submit_button = Button(root, text='Расcчитать',width=30,bg=calc_button_color_1,fg='white', height=2, activebackground=calc_button_color_2,
                command= lambda: self.submit_button_wrapper(
                    KeepTrackXYEntryLabel,
                    main_entry_list_vars,
                    current_frame,
                    inp_faza_razshep,
                    inp_type_razchep,
                    excel_filepath,
                    mat_prop_vars_list,
                    xy_var_list,
                    pop_sech_edin_izmer,
                    inter_izmer_edin_izmer,
                    dlina_edin_izmer,
                    gamma_edin_izmer,
                    naprzh_linii_edin_izmer,
            ))
        self.submit_button.place(x=1050,y=630)
        self.submit_button["state"] = "disabled"
        globals()["submit_button"] = self.submit_button

    def submit_button_wrapper(self, *args):

        answer = messagebox.askyesno(title="Расчет", message="Вы уверены что хотите произвести раcчет по указанным данным?")
        if answer == False: return;

        def submit_button():

            # print("Submitted values: ")

            KeepTrackXYEntryLabel, main_entry_list_vars, current_frame, inp_faza_razshep, inp_type_razchep, excel_filepath, mat_prop_vars_list, xy_var_list, pop_sech_edin_izmer, inter_izmer_edin_izmer, dlina_edin_izmer, gamma_edin_izmer, naprzh_linii_edin_izmer = args
            faza_razshep, type_razchep, = bool(inp_faza_razshep.get()), bool(inp_type_razchep.get())

            # Extracting and preprocessing user input
            try:
                excel_filepath = excel_filepath.get()
                excel_filepath_os = Path(excel_filepath)

                name_podstans = main_entry_list_vars[0].get().strip()
                name_prisoedin = main_entry_list_vars[1].get().strip()
                kolich_prisoed = float(main_entry_list_vars[2].get())
                dlina_linii = float(main_entry_list_vars[3].get())
                kolich_izmer = float(main_entry_list_vars[4].get())
                interval_izmer = float(main_entry_list_vars[5].get())
                kolich_garmonik = float(main_entry_list_vars[6].get())
                inp_napryazhen_linii = float(main_entry_list_vars[7].get())


                if dlina_edin_izmer.get() == "м":
                    dlina_linii = dlina_linii/1000
                if inter_izmer_edin_izmer.get() == 'сек':
                    interval_izmer = interval_izmer/60
                if naprzh_linii_edin_izmer.get() == "В":
                    inp_napryazhen_linii = inp_napryazhen_linii/1000
                    print(inp_napryazhen_linii)


            except ValueError:
                messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных воздушной линии")
                main_entry_list_vars = None
                return

            list_of_xys, getted_list_xys, floated_list_xys, material_properties, floated_list_matprop = [], [], [], [], []


            # Module for XandY data
            if current_frame != None:
                if current_frame == 1:
                    tx, ty, ax, ay, bx, by, cx, cy =  xy_var_list
                    list_of_xys = [tx, ty, ax, ay, bx, by, cx, cy]
                if current_frame == 2:
                    tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y =  xy_var_list
                    list_of_xys = [tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y]
                if current_frame == 3:
                    t1x, t1y, t2x, t2y, ax, ay, bx, by, cx, cy =  xy_var_list
                    list_of_xys = [t1x, t1y, t2x, t2y, ax, ay, bx, by, cx, cy]
                getted_list_xys = list(map(lambda e: e.get(), list_of_xys))
                try:
                    floated_list_xys = list(map(lambda e: float(e), getted_list_xys))
                except ValueError:
                    messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для координат фазных проводов и троса")
                    floated_list_xys = []
                    return

            # Module for Material Prop. data
            getted_list_matprop = list(map(lambda e: e.get(), mat_prop_vars_list))

            # Unit conversions happens here
            try:
                floated_list_matprop = list(map(lambda e: float(e), getted_list_matprop))
                if gamma_edin_izmer.get() == "кСм/м":
                    floated_list_matprop[1] = floated_list_matprop[1]*10**3
                    floated_list_matprop[4] = floated_list_matprop[4]*10**3
                if gamma_edin_izmer.get() == "См/м":
                    floated_list_matprop[1] = floated_list_matprop[1]*10**6
                    floated_list_matprop[4] = floated_list_matprop[4]*10**6

                if pop_sech_edin_izmer.get() == "м\u00b2":
                    floated_list_matprop[2] = floated_list_matprop[2]*10**6
                    floated_list_matprop[5] = floated_list_matprop[5]*10**6
                if pop_sech_edin_izmer.get() == "см\u00b2":
                    floated_list_matprop[2] = floated_list_matprop[2]*100
                    floated_list_matprop[5] = floated_list_matprop[5]*100

            except ValueError:
                messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для характеристик материала фазных проводов и троса")
                floated_list_matprop = []
                return


            # print("MatProps: ", floated_list_matprop)
            # print("XYs: ", floated_list_xys)

            if floated_list_xys and floated_list_matprop and main_entry_list_vars and excel_filepath:
                # print("Data is full")

                # Start Progress Bar
                progress_pivot = (770,645)
                progress_bar = Progressbar(root, orient=HORIZONTAL,length=218,  mode='indeterminate')
                progress_bar.place(x=progress_pivot[0], y=progress_pivot[1])
                progress_bar_label = Label(root, text="Обработка данных...", width=23, font=("bold", 12), bg=main_label_color, fg=main_text_color)
                progress_bar_label.place(x=progress_pivot[0], y=progress_pivot[1]-30)
                self.submit_button["state"] = "disabled"
                progress_bar.start()

                # Clear all entries
                for entry in KeepTrackXYEntryLabel.overall_whole_entries:
                    try:
                        entry.delete(0, 'end')
                    except:
                        print("Something went wrong!")
                finishing_part(name_podstans)
                final_message = f"Результаты расчетов записаны в папке 'Результаты_{name_podstans}'"


                sleep(5) # EMULATING CALCULATION PROCESS
                progress_bar.stop()
                self.submit_button["state"] = "normal"
                progress_bar_label.destroy()
                progress_bar.destroy()
                messagebox.showinfo(title="Расчет завершен!", message="Данные успешно обработаны!", detail=final_message)
                # os.system("taskkill /F /IM python3.8.exe /T")
            else:
                # print("Not enough input data")
                messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных")
                return

        threading.Thread(target=submit_button).start()



def resource_path(relative_path):
    try:
        base_path = Path(".")
    except Exception:
        base_path = sys._MEIPASS
    return base_path / relative_path


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


def validate_numbers(index, numbers):
    # print("Modification at index " + index)
    return globals()["pattern"].match(numbers) is not None


def draw_mater_cable_grid(x, y):

    ropes = ["Фаза", "Трос"]
    chars = ["Маг.проницаемость (\u03bc)", "Уд.проводимость (\u03C3)", "Попер.сечение (S)"]

    types_cables = ["inp_faza_", "inp_tros_"]
    mater_charac = ["mag", "gam", "pop"]

    mat_prop_vars_list = []

    for provod in types_cables:
        for character in mater_charac:
            exec(provod+character + "=StringVar()")
            mat_prop_vars_list.append(eval(provod+character))

    globals()["mat_prop_vars_list"] = mat_prop_vars_list

    for i, cha in enumerate(chars, start=1):
        label_xy = Label(root, text=cha,
            width=21,font=("bold", 11), borderwidth=2, relief="groove", bg=label_color, fg=text_color)
        label_xy.place(x=(i*184)+x,y=y-20)

    for i, prov in enumerate(ropes, start=0):
        label_frame = Label(root, text=prov,
            width=20, font=("bold", 11), borderwidth=2, relief="groove", bg=label_color, fg=text_color)
        label_frame.place(x=x,y=(i*21)+y)


    for i, provod in enumerate(types_cables, start=0):
        for j, character in enumerate(mater_charac, start=1):
            pattern = re.compile(r'^([\.\d]*)$')
            globals()["pattern"] = pattern
            vcmd = (root.register(validate_numbers), "%i", "%P")
            entry_cell = KeepTrackXYEntryLabel("Entry", "General", root, textvariable=eval(provod+character), width=31, validate="key", validatecommand=vcmd, bg=entry_color)
            entry_cell.place(x=(j*185)+x,y=y+(i*21), height=21);

class KeepTrackXYEntryLabel(Entry, Label):

    current_frame = None
    all_entries = {
        "Anker1": [],
        "Anker2": [],
        "Prom3": [],
    }
    all_labels = {
        "Anker1": [],
        "Anker2": [],
        "Prom3": [],
    }
    overall_whole_entries = []

    def __init__(self, ctype, flag, *args,**kwargs):

        if ctype == "Entry":
            Entry.__init__(self, *args,**kwargs)
            self.my_add_entry(self, flag, ctype)
        if ctype == "Label":
            Label.__init__(self, *args,**kwargs)
            self.my_add_entry(self, flag, ctype)

    @classmethod
    def my_add_entry(cls, ent, flag, ctype):
        if flag != "General":
            if ctype == "Entry":
                cls.all_entries[flag].append(ent)
                cls.overall_whole_entries.append(ent)
            if ctype == "Label":
                cls.all_labels[flag].append(ent)
        else:
            cls.overall_whole_entries.append(ent)

    @classmethod
    def draw_xy_grid(cls, frame_type, x, y):

        xy_labels = ["X", "Y"]
        frame_label_dict = {
            "Prom3" : ["T1","T2","A","B","C"],
            "Anker1" : ["T","A","B","C"],
            "Anker2" : ["T","A1","B1","C1","A2","B2","C2"],
        }
        xy_var_list = []

        for ank_lab in frame_label_dict[frame_type]:
            for xy in xy_labels:
                inp_var = (ank_lab+xy).lower()
                exec(inp_var + "=StringVar()")
                xy_var_list.append(eval(inp_var))

        globals()["xy_var_list"] = xy_var_list

        for i, xy in enumerate(xy_labels, start=1):
            label_xy = KeepTrackXYEntryLabel("Label", frame_type, root, text=xy,width=10,
                 height=20, font=("bold", 11), borderwidth=2, relief="groove", bg=label_color, fg=text_color)
            label_xy.config(relief = RAISED, height = 1)
            label_xy.place(x=(i*97)+x,y=y-22)

        for i, anker in enumerate(frame_label_dict[frame_type], start=0):
            label_frame = KeepTrackXYEntryLabel("Label", frame_type, root, text=anker,width=10, height=20, font=("bold", 11), borderwidth=2, relief="groove", bg=label_color, fg=text_color)
            label_frame.config(relief = RAISED, height = 1)
            label_frame.place(x=x,y=(i*21)+y)


        for i, lab in enumerate(frame_label_dict[frame_type], start=0):
            for j, xy in enumerate(xy_labels, start=1):
                pattern = re.compile(r'^([\.\d]*)$')
                globals()["pattern"] = pattern
                vcmd = (root.register(validate_numbers), "%i", "%P")
                entry_cell = KeepTrackXYEntryLabel("Entry", frame_type, root, width=16, textvariable=eval((lab+xy).lower()), validate="key", validatecommand=vcmd, bg=entry_color)
                entry_cell.place(x=(j*96)+x,y=y+(i*21), height=21);

    @classmethod
    def destroy_xy_grid(cls, frame_type):
        for entry in cls.all_entries[frame_type]:
            entry.destroy()
        for label in cls.all_labels[frame_type]:
            label.destroy()


def on_click(event):
    # print("Invoked!")

    # globals()["main1_entry"].configure(state=NORMAL)
    globals()["main1_entry"].delete(0, END)

    # make the callback only work once
    globals()["main1_entry"].unbind('<Button-1>', globals()["on_click_id"])

def draw_main_block(x, y):

    # Main label
    label_main = Label(root, text="Расчет добавочных потерь на ЛЭП",width=30,font=("underline", 25), bg=label_color, fg=text_color)
    label_main.place(x=x-70,y=y-77)

    main_entry_list_names = [
        ("inp_name_podstans", "Наименование подстанции"),
        ("inp_name_prisoedin", "Наименования присоед-ии."),
        ("inp_kolich_prisoed", "Количество присоед."),
        ("inp_dlina_linii", "Длина линии"),
        ("inp_kolich_izmer", "Количество измерении."),
        ("inp_interval_izmer", "Интервал измерении"),
        ("inp_kolich_garmonik", "Количество гармоник"),
        ("inp_napryazhen_linii", "Напряжение линии"),
    ]
    main_entry_list_vars = []

    for main_var, _ in main_entry_list_names:
        exec(main_var + "=StringVar()")
        main_entry_list_vars.append(eval(main_var))

    globals()["main_entry_list_vars"] = main_entry_list_vars

    for i in range(len(main_entry_list_names)):
        eng, rus = main_entry_list_names[i]

        main_label = Label(root, text=rus, width=23, font=("bold", 12), bg=main_label_color, fg=main_text_color)
        main_label.place(x=x, y=y+i*50)

        if eng == "inp_name_podstans":
            main0_entry = KeepTrackXYEntryLabel("Entry", "General", root, textvariable=main_entry_list_vars[i], bg=entry_color)
            main0_entry.place(x=x+220, y=100+i*50)

        elif eng == "inp_name_prisoedin":
            main1_entry = KeepTrackXYEntryLabel("Entry", "General", root, textvariable=main_entry_list_vars[i], bg=entry_color)
            main1_entry.place(x=x+220, y=100+i*50)
            main1_entry.insert(0, 'Каждую через ","')

            main1_entry.bind('<Button-1>', on_click)
            globals()["main1_entry"] = main1_entry
            on_click_id = main1_entry.bind('<Button-1>', on_click)
            globals()["on_click_id"] = on_click_id
        else:
            pattern = re.compile(r'^([\.\d]*)$')
            globals()["pattern"] = pattern
            vcmd = (root.register(validate_numbers), "%i", "%P")
            main_entry = KeepTrackXYEntryLabel("Entry", "General", root, textvariable=main_entry_list_vars[i], validate="key", validatecommand=vcmd, bg=entry_color)
            main_entry.place(x=x+220, y=100+i*50)


def type_razche(root, inp_faza_razshep):
    global b2, b3, b4 #treug_but_var, kvadrat_but_var, dwa_but_var
    raz_state = inp_faza_razshep.get()

    if raz_state == True:

        b2 = Radiobutton(root, text="(2) Двойная",padx = 5, font=("bold", 11), variable=inp_type_razchep,
            value=1, bg=main_label_color, fg=main_text_color)
        b2.place(x=395,y=525)
        b3 = Radiobutton(root, text="(3) Прав.треуголь",padx = 20, font=("bold", 11), variable=inp_type_razchep,
            value=2, bg=main_label_color, fg=main_text_color)
        b3.place(x=380,y=555)
        b4 = Radiobutton(root, text="(4) Квадрат",padx = 20, font=("bold", 11), variable=inp_type_razchep,
            value=3, bg=main_label_color, fg=main_text_color)
        b4.place(x=380,y=590)

    if raz_state == False:
        b2.destroy()
        b3.destroy()
        b4.destroy()



def frame_type(root, radiovar, canvas, coord):

    global current_frame, submit_button

    type_of_frame = radiovar.get()
    canvas.delete("all")
    submit_button["state"] = NORMAL

    if current_frame == 1:
        KeepTrackXYEntryLabel.destroy_xy_grid("Anker1")
    if current_frame == 2:
        KeepTrackXYEntryLabel.destroy_xy_grid("Anker2")
    if current_frame == 3:
        KeepTrackXYEntryLabel.destroy_xy_grid("Prom3")


    if type_of_frame == 1:
        img = ImageTk.PhotoImage(Image.open(resource_path(Path("pics/VL1.jpg"))))
        flip_button["state"] = NORMAL
        KeepTrackXYEntryLabel.draw_xy_grid("Anker1", *coord)
        current_frame = type_of_frame


    if type_of_frame == 2:
        flip_button["state"] = "disable"
        img = ImageTk.PhotoImage(Image.open(resource_path(Path("pics/VL2.jpg"))))
        KeepTrackXYEntryLabel.draw_xy_grid("Anker2", *coord)
        current_frame = type_of_frame

    if type_of_frame == 3:
        flip_button["state"] = "disable"
        img = ImageTk.PhotoImage(Image.open(resource_path(Path("pics/VL3.jpg"))))
        KeepTrackXYEntryLabel.draw_xy_grid("Prom3", *coord)
        current_frame = type_of_frame

    canvas.create_image(1, 1, anchor=NW, image=img)
    root.mainloop()


# Image mirroring function
def flip_function(rood, radiovar, canvas):
    type_of_frame = radiovar.get()
    canvas.delete("all")

    global flip_state

    if flip_state == type_of_frame:
        if flip_state == 1 and type_of_frame == 1:
            img = ImageTk.PhotoImage(Image.open(resource_path(Path("pics/VL1.jpg"))))
            flip_state = 2
    else:
        if type_of_frame == 1:
            img = ImageTk.PhotoImage(Image.open(resource_path(Path("pics/VL1_mir.jpg"))))
            flip_state = type_of_frame

    canvas.create_image(1, 1, anchor=NW, image=img)
    root.mainloop()



if __name__ == "__main__":
    try:
        # Main Configurations
        root = Tk()

        width, height = root.winfo_screenwidth(), root.winfo_screenheight()

        root.geometry('%dx%d+0+0' % (width,height))

        root.configure(bg=window_color)
        root.title("Расчет добавочных потерь на ЛЭП")

        # Variables of the GUI
        frame = IntVar()
        inp_faza_razshep = BooleanVar()
        inp_type_razchep = BooleanVar()

        flip_state = None
        current_frame = None


        # Main label
        draw_main_block(80, 97)

        # Grid of material characteristic labels
        draw_mater_cable_grid(600, 463)

        # Module for scaling options
        dlina_edin_izmer = StringVar(root)
        dlina_edin_izmer.set("км") # default value
        dlina_opt_box = OptionMenu(root, dlina_edin_izmer, "м", "км")
        dlina_opt_box.pack()
        dlina_opt_box.place(x=450, y=245)


        inter_izmer_edin_izmer = StringVar(root)
        inter_izmer_edin_izmer.set("мин") # default value
        inter_izmer_opt_box = OptionMenu(root, inter_izmer_edin_izmer, "сек", "мин")
        inter_izmer_opt_box.pack()
        inter_izmer_opt_box.place(x=450, y=345)


        naprzh_linii_edin_izmer = StringVar(root)
        naprzh_linii_edin_izmer.set("кВ") # default value
        naprzh_linii_opt_box = OptionMenu(root, naprzh_linii_edin_izmer, "В", "кВ")
        naprzh_linii_opt_box.pack()
        naprzh_linii_opt_box.place(x=450, y=445)


        gamma_edin_izmer = StringVar(root)
        gamma_edin_izmer.set("МСм/м") # default value
        gamma_opt_box = OptionMenu(root, gamma_edin_izmer, "См/м", "кСм/м", "МСм/м")
        gamma_opt_box.pack()
        gamma_opt_box.place(x=1015, y=510)


        pop_sech_edin_izmer = StringVar(root)
        pop_sech_edin_izmer.set("мм\u00b2") # default value
        pop_sech_opt_box = OptionMenu(root, pop_sech_edin_izmer, "мм\u00b2", "см\u00b2", "м\u00b2")
        pop_sech_opt_box.pack()
        pop_sech_opt_box.place(x=1215, y=510)


        # IMAGE DISPLAYING ON WINDOWS
        canvas = Canvas(root, width = 281, height = 313, bg='white')
        canvas.pack(expand=YES, fill=BOTH)
        canvas.place(x=650,y=30)


        # Frame type radio buttons
        label_frame = Label(root, text="Тип опоры",width=20,font=("bold", 13), bg=label_color, fg=text_color)
        label_frame.place(x=60,y=495)


        xy_grid_coords = (980, 60)

        rad_but_1 = Radiobutton(root, text="Промеж.одноцепная",padx = 5, font=("bold", 11), variable=frame,
            value=1, command = lambda: frame_type(root, frame, canvas, xy_grid_coords), bg=main_label_color, fg=main_text_color)
        rad_but_2 = Radiobutton(root, text="Промеж.двухцепная",padx = 20, font=("bold", 11), variable=frame,
            value=2, command = lambda: frame_type(root, frame, canvas, xy_grid_coords), bg=main_label_color, fg=main_text_color)
        rad_but_3 = Radiobutton(root, text="Промеж.портальная",padx = 20, font=("bold", 11), variable=frame,
            value=3, command = lambda: frame_type(root, frame, canvas, xy_grid_coords), bg=main_label_color, fg=main_text_color)
        rad_but_1.place(x=65, y=525)
        rad_but_2.place(x=50, y=560)
        rad_but_3.place(x=50, y=595)

        label_frame = Label(root, text="Фаза",width=20,font=("bold", 13), bg=label_color, fg=text_color)
        label_frame.place(x=250,y=495)

        label_faza_raz = Label(root, text="pасщеплена",width=11,font=("bold", 13), bg=label_color, fg=text_color)
        label_faza_raz.place(x=425,y=495)

        razshep_chek_but = Checkbutton(root, text="",
                                     variable=inp_faza_razshep,
                                     onvalue = True, offvalue = False,
                                     command = lambda: type_razche(root, inp_faza_razshep),
                                    indicatoron=1, bg=label_color)
        razshep_chek_but.place(x=400, y=495)


        # Flip button of frame type
        flip_button = Button(root,text="Отзеркалить",font=40,bg=label_color,fg='white',command=lambda:flip_function(root, frame, canvas))
        flip_button.place(x=735,y=350)
        flip_button["state"] = "disabled"


        # FILE INPUT FOR EXCEL FILE
        def fileinput():
            filename = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls"), ("ALL","*.*")])
            excel_filepath.insert(END, filename) # add this


        label_excel = Label(root, text="Путь к файлу:",width=20,font=("bold", 12), bg=label_color, fg=text_color)
        label_excel.place(x=55,y=650)

        excel_filepath = KeepTrackXYEntryLabel("Entry", "General", root,font=30, bg=entry_color)
        excel_filepath.grid(row=2,column=6)
        excel_filepath.place(x=245,y=650)

        select_button_excel = Button(root,text="Выбрать",font=40,bg=label_color,fg='white',command=fileinput).place(x=430,y=645)

        submit_instance = SubmitProgressBar()



        # Final stage of GUI. Displaying!
        root.mainloop()
    finally:
        os.system("taskkill /F /IM python3.8.exe /T")

    # print("Power Loss Calculator Has Been Successfully Created")



